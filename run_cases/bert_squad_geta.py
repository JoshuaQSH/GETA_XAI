"""
BERT SQuAD GETA Experiment
==========================

Train BERT on SQuAD using the GETA optimizer for joint pruning and quantization.

Usage:
    python bert_squad_geta.py --epochs 3 --sparsity 0.7
    python bert_squad_geta.py --epochs 1  # Quick test

Experimental Setup:
- Model: BERT-base-uncased (2 layers for quick testing)
- Dataset: SQuAD v2.0
- Task: Question Answering
- Bit width reduction: b_r = 2
- Bit width range: [b_l, b_u] = [4, 16]
- Exponential: t = 1
- Sparsity: 0.7
- Total Epochs: 3
- Projection Periods B: 5
- Projection Steps K_b: 1 epoch
- Pruning Periods P: 10
- Pruning Steps K_p: 1 epoch
- Optimizer: AdamW (1e-3)
"""

import json
import os
import random
import sys
import time
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from only_train_once import OTO
from only_train_once.optimizer import GETA

from run_cases.utils import (
    get_base_parser,
    args_to_config,
    ExperimentConfig,
    ExperimentResults,
    save_results_to_csv,
    print_results_summary,
    print_epoch_summary,
    print_config,
    get_timestamp,
    generate_experiment_name,
    ensure_output_dir,
)


class SquadDataset(Dataset):
    """
    Full SQuAD v2.0 dataset for Question Answering.
    
    Properly handles:
    - Tokenization with question-context pairs
    - Answer span computation (start/end token positions)
    - Unanswerable questions (SQuAD 2.0)
    - Long context truncation with stride
    """

    def __init__(self, tokenizer, file_path, max_length=384, doc_stride=128, max_samples=None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            file_path: Path to SQuAD JSON file
            max_length: Maximum sequence length
            doc_stride: Stride for splitting long documents
            max_samples: Optional limit on number of samples (None for full dataset)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.features = []
        
        print(f"Loading SQuAD dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process all QA pairs
        count = 0
        for article in tqdm(data['data'], desc="Processing articles"):
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    question = qa['question']
                    qa_id = qa['id']
                    is_impossible = qa.get('is_impossible', False)
                    
                    # Get answer info
                    if is_impossible or not qa['answers']:
                        # Unanswerable question
                        answer_text = ""
                        answer_start_char = 0
                    else:
                        answer = qa['answers'][0]
                        answer_text = answer['text']
                        answer_start_char = answer['answer_start']
                    
                    # Tokenize question and context
                    tokenized = self.tokenizer(
                        question,
                        context,
                        max_length=max_length,
                        truncation='only_second',  # Only truncate context
                        stride=doc_stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    
                    # Process each chunk (for long contexts)
                    sample_mapping = tokenized.pop('overflow_to_sample_mapping', None)
                    offset_mapping = tokenized.pop('offset_mapping')
                    
                    num_chunks = tokenized['input_ids'].shape[0]
                    
                    for chunk_idx in range(num_chunks):
                        input_ids = tokenized['input_ids'][chunk_idx]
                        attention_mask = tokenized['attention_mask'][chunk_idx]
                        token_type_ids = tokenized['token_type_ids'][chunk_idx]
                        offsets = offset_mapping[chunk_idx]
                        
                        # Find answer span in tokens
                        start_position = 0
                        end_position = 0
                        
                        if not is_impossible and answer_text:
                            answer_end_char = answer_start_char + len(answer_text)
                            
                            # Find sequence IDs (0 for question, 1 for context, None for special tokens)
                            sequence_ids = tokenized.sequence_ids(chunk_idx)
                            
                            # Find context start and end in tokens
                            context_start = None
                            context_end = None
                            for idx, seq_id in enumerate(sequence_ids):
                                if seq_id == 1:
                                    if context_start is None:
                                        context_start = idx
                                    context_end = idx
                            
                            if context_start is not None and context_end is not None:
                                # Check if answer is within this chunk
                                chunk_start_char = offsets[context_start][0]
                                chunk_end_char = offsets[context_end][1]
                                
                                if answer_start_char >= chunk_start_char and answer_end_char <= chunk_end_char:
                                    # Find token positions
                                    for idx in range(context_start, context_end + 1):
                                        if offsets[idx][0] <= answer_start_char < offsets[idx][1]:
                                            start_position = idx
                                        if offsets[idx][0] < answer_end_char <= offsets[idx][1]:
                                            end_position = idx
                                            break
                        
                        self.features.append({
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'token_type_ids': token_type_ids,
                            'start_positions': torch.tensor(start_position, dtype=torch.long),
                            'end_positions': torch.tensor(end_position, dtype=torch.long),
                            'qa_id': qa_id,
                            'is_impossible': is_impossible
                        })
                        
                        count += 1
                        if max_samples is not None and count >= max_samples:
                            break
                    if max_samples is not None and count >= max_samples:
                        break
                if max_samples is not None and count >= max_samples:
                    break
            if max_samples is not None and count >= max_samples:
                break
        
        print(f"Loaded {len(self.features)} features from {count} QA pairs")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
            'input_ids': feature['input_ids'],
            'attention_mask': feature['attention_mask'],
            'token_type_ids': feature['token_type_ids'],
            'start_positions': feature['start_positions'],
            'end_positions': feature['end_positions']
        }


def get_squad_loaders(config: ExperimentConfig, tokenizer, max_train_samples=None, max_eval_samples=None) -> tuple:
    """
    Load full SQuAD dataset and return train/test loaders.
    
    Args:
        config: Experiment configuration
        tokenizer: HuggingFace tokenizer
        max_train_samples: Optional limit on training samples (None for full dataset)
        max_eval_samples: Optional limit on eval samples (None for full dataset)
    """
    
    train_dataset = SquadDataset(
        tokenizer,
        os.path.join(config.dataset_root, 'SQuAD', 'train-v2.0.json'),
        max_length=384,
        doc_stride=128,
        max_samples=max_train_samples
    )

    test_dataset = SquadDataset(
        tokenizer,
        os.path.join(config.dataset_root, 'SQuAD', 'dev-v2.0.json'),
        max_length=384,
        doc_stride=128,
        max_samples=max_eval_samples
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return trainloader, testloader, train_dataset


def create_quantized_bert(
    device: str = 'cuda:0',
    model_name: str = 'bert-base-uncased',
    qa_checkpoint: str = '',
    num_hidden_layers: int = 2,
) -> tuple:
    """Create quantized BERT model and dummy input."""
    from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode

    if qa_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(qa_checkpoint)
        model = AutoModelForQuestionAnswering.from_pretrained(qa_checkpoint)
        if (
            num_hidden_layers > 0
            and hasattr(model, "bert")
            and hasattr(model.bert, "encoder")
            and hasattr(model.bert.encoder, "layer")
            and num_hidden_layers < len(model.bert.encoder.layer)
        ):
            model.bert.encoder.layer = torch.nn.ModuleList(
                list(model.bert.encoder.layer[:num_hidden_layers])
            )
            model.config.num_hidden_layers = num_hidden_layers
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name, num_hidden_layers=num_hidden_layers)
        model = AutoModelForQuestionAnswering.from_config(config)

    model = model_to_quantize_model(
        model,
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION,
        q_m_init=1.0,  # Use default q_m to avoid overflow
    )

    # Create dummy input
    text = "This is a test question. " * 50
    inputs = tokenizer(text, return_tensors='pt', max_length=384, truncation=True, padding='max_length')
    dummy_input = (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

    return model.to(device), tuple(t.to(device) for t in dummy_input), tokenizer


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_qa_model(model, testloader, device: str = 'cuda') -> tuple:
    """
    Evaluate QA model with proper metrics.
    
    Returns:
        Tuple of (exact_match_score, f1_score, loss)
    """
    model.eval()
    total_loss = 0.0
    total_em = 0
    total_f1 = 0.0
    total_samples = 0
    
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in testloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Compute loss
            start_loss = loss_fn(outputs.start_logits, start_positions)
            end_loss = loss_fn(outputs.end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            total_loss += loss.item() * input_ids.size(0)
            
            # Compute predictions
            pred_start = torch.argmax(outputs.start_logits, dim=1)
            pred_end = torch.argmax(outputs.end_logits, dim=1)
            
            # Exact match: both start and end must be correct
            em = ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
            total_em += em
            
            # Token-level F1 (simplified)
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # Get predicted and true spans
                pred_s, pred_e = pred_start[i].item(), pred_end[i].item()
                true_s, true_e = start_positions[i].item(), end_positions[i].item()
                
                if pred_e < pred_s:
                    pred_s, pred_e = pred_e, pred_s
                
                # Compute overlap
                overlap_start = max(pred_s, true_s)
                overlap_end = min(pred_e, true_e)
                
                if overlap_start <= overlap_end:
                    overlap_len = overlap_end - overlap_start + 1
                    pred_len = pred_e - pred_s + 1
                    true_len = true_e - true_s + 1
                    
                    precision = overlap_len / pred_len if pred_len > 0 else 0
                    recall = overlap_len / true_len if true_len > 0 else 0
                    
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
                else:
                    f1 = 0.0
                
                total_f1 += f1
            
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    em_score = 100.0 * total_em / total_samples if total_samples > 0 else 0.0
    f1_score = 100.0 * total_f1 / total_samples if total_samples > 0 else 0.0
    
    return em_score, f1_score, avg_loss


def train_geta(
    trainloader,
    testloader,
    model,
    oto,
    tokenizer,
    config: ExperimentConfig,
) -> ExperimentResults:
    """
    Train BERT with GETA optimizer.

    Args:
        trainloader: Training data loader
        testloader: Test data loader
        model: Quantized BERT model
        oto: OTO instance
        tokenizer: BERT tokenizer
        config: Experiment configuration

    Returns:
        ExperimentResults with all metrics
    """

    # Calculate steps based on epochs
    steps_per_epoch = len(trainloader)
    projection_steps = config.projection_epochs * steps_per_epoch
    pruning_steps = config.pruning_epochs * steps_per_epoch
    start_pruning_step = projection_steps  # Start pruning after projection

    # Create GETA optimizer
    param_groups = list(oto._graph.get_param_groups())
    
    # Deduplicate parameters across all groups to avoid PyTorch optimizer error
    # This is needed for BERT models where some params appear multiple times
    global_seen_ids = set()
    for pg in param_groups:
        unique_indices = []
        for i, p in enumerate(pg['params']):
            p_id = id(p)
            if p_id not in global_seen_ids:
                global_seen_ids.add(p_id)
                unique_indices.append(i)
        
        if len(unique_indices) < len(pg['params']):
            # Filter to unique params only
            pg['params'] = [pg['params'][i] for i in unique_indices]
            pg['p_names'] = [pg['p_names'][i] for i in unique_indices]
            pg['p_transform'] = [pg['p_transform'][i] for i in unique_indices]
            if 'op_names' in pg:
                pg['op_names'] = [pg['op_names'][i] for i in unique_indices]
            if 'node_ids' in pg:
                pg['node_ids'] = [pg['node_ids'][i] for i in unique_indices]
    
    # Remove empty groups
    param_groups = [pg for pg in param_groups if len(pg['params']) > 0]

    optimizer = GETA(
        params=param_groups,
        variant="adamw",
        lr=config.lr,
        lr_quant=config.lr_quant,
        first_momentum=0.9,
        weight_decay=config.weight_decay,
        target_group_sparsity=config.target_sparsity,
        start_projection_step=config.start_projection_step,
        projection_periods=config.projection_periods,
        projection_steps=projection_steps,
        start_pruning_step=start_pruning_step,
        pruning_periods=config.pruning_periods,
        pruning_steps=pruning_steps,
        bit_reduction=config.bit_reduction,
        min_bit_wt=config.min_bit,
        max_bit_wt=config.max_bit,
        device=config.device,
    )

    # Store optimizer in OTO
    oto._optimizer = optimizer

    # Training setup
    # model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=0.1
    )
    
    # Get initial MACs and BOPs
    print("Computing initial MACs and BOPs...")
    full_macs_info = oto.compute_macs(in_million=True)
    full_bops_info = oto.compute_bops(in_million=True)
    full_macs = full_macs_info["total"]
    full_bops = full_bops_info["total"]
    print(f"Full MACs: {full_macs:.2f} M")
    print(f"Full BOPs: {full_bops:.2f} M")
    
    # Training loop
    print(f"\nTraining BERT with GETA for {config.epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Projection steps: {projection_steps}, Pruning steps: {pruning_steps}")
    print("-" * 80)

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Training
        # correct = 0
        # total = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            token_type_ids = batch['token_type_ids'].to(config.device)
            start_positions = batch['start_positions'].to(config.device)
            end_positions = batch['end_positions'].to(config.device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Calculate loss for start and end positions
            start_loss = criterion(outputs.start_logits, start_positions)
            end_loss = criterion(outputs.end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix({
                'loss': f"{epoch_loss/(batch_idx+1):.3f}"
            })

        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate and print epoch summary
        avg_loss = epoch_loss / len(trainloader)
        val_em, val_f1, val_loss = evaluate_qa_model(model, testloader, config.device)
        metrics = optimizer.compute_metrics()
        
        # Use F1 as primary metric for epoch summary
        print_epoch_summary(epoch, config.epochs, avg_loss, val_em, val_f1, metrics)

    total_time = time.time() - start_time
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    final_em, final_f1, final_loss = evaluate_qa_model(model, testloader, config.device)
    final_metrics = optimizer.compute_metrics()
    
    # Compute compressed MACs and BOPs
    compressed_macs_info = oto.compute_macs(in_million=True)
    compressed_bops_info = oto.compute_bops(in_million=True)
    compressed_macs = compressed_macs_info["total"]
    compressed_bops = compressed_bops_info["total"]
    
    print(f"Compressed MACs: {compressed_macs:.2f} M")
    print(f"Compressed BOPs: {compressed_bops:.2f} M")
    print(f"Final EM: {final_em:.2f}%, F1: {final_f1:.2f}%")
    
    # Create results
    results = ExperimentResults(
        model_name='BERT-SQuAD',
        optimizer_type='GETA',
        epochs=config.epochs,
        target_sparsity=config.target_sparsity,
        attribution_method='',
        attribution_weight=0.0,
        full_macs=full_macs,
        full_bops=full_bops,
        compressed_macs=compressed_macs,
        compressed_bops=compressed_bops,
        final_top1_accuracy=final_em,  # EM score
        final_top5_accuracy=final_f1,  # F1 score
        total_param_norm=final_metrics.norm_params,
        group_sparsity=final_metrics.group_sparsity,
        num_important_groups=final_metrics.num_important_groups,
        num_redundant_groups=final_metrics.num_redundant_groups,
        num_zero_groups=final_metrics.num_zero_groups,
        total_training_time=total_time,
        timestamp=get_timestamp(),
    )

    return results, model, oto


def main():
    """Main function to run GETA experiment."""

    # Parse arguments
    parser = get_base_parser('BERT SQuAD GETA Experiment')
    parser.add_argument(
        '--model-name',
        type=str,
        default='bert-base-uncased',
        help='Base HuggingFace model name when building a fresh QA model.',
    )
    parser.add_argument(
        '--qa-checkpoint',
        type=str,
        default='',
        help='Optional pretrained QA checkpoint to fine-tune instead of a fresh model.',
    )
    parser.add_argument(
        '--num-hidden-layers',
        type=int,
        default=2,
        help='Number of hidden layers when building a fresh QA model from config.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2027,
        help='Random seed for model init and data order.',
    )
    args = parser.parse_args()
    seed_all(args.seed)
    config = args_to_config(args, is_xai=False)

    # Print configuration
    print("\n" + "=" * 70)
    print("BERT SQuAD GETA Experiment")
    print("=" * 70)
    print_config(config, 'GETA')

    # Ensure output directory exists
    ensure_output_dir(config)

    # Step 1: Create model
    print("[Step 1] Creating quantized BERT model...")
    model, dummy_input, tokenizer = create_quantized_bert(
        config.device,
        model_name=args.model_name,
        qa_checkpoint=args.qa_checkpoint,
        num_hidden_layers=args.num_hidden_layers,
    )

    # Step 2: Initialize OTO
    print("[Step 2] Initializing OTO framework...")
    oto = OTO(model=model, dummy_input=dummy_input, strict_out_nodes=False)

    # Exclude embeddings from pruning
    oto.mark_unprunable_by_param_names(["bert.embeddings.word_embeddings.weight"])

    # Step 3: Load dataset
    print("[Step 3] Loading SQuAD dataset...")
    max_train = getattr(args, 'max_train_samples', None)
    max_eval = getattr(args, 'max_eval_samples', None)
    trainloader, testloader, trainset = get_squad_loaders(
        config, tokenizer, 
        max_train_samples=max_train,
        max_eval_samples=max_eval
    )
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testloader.dataset)}")

    # Step 4: Train with GETA
    print("\n[Step 4] Training with GETA optimizer...")
    results, trained_model, oto = train_geta(
        trainloader, testloader, model, oto, tokenizer, config
    )

    # Print and save results
    print_results_summary(results)

    # Generate experiment name and save results
    exp_name = generate_experiment_name('geta', config, model_name='bert_squad')
    csv_path = os.path.join(config.output_dir, f"{exp_name}_results.csv")
    save_results_to_csv(results, csv_path)

    print("\n" + "=" * 70)
    print("GETA Experiment Complete!")
    print("=" * 70 + "\n")

    return results, trained_model, oto


if __name__ == "__main__":
    main()
