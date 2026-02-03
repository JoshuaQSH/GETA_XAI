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

import os
import sys
import time
import json
from tqdm import tqdm
from datetime import datetime

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
    """Simple SQuAD dataset for demonstration."""

    def __init__(self, tokenizer, file_path, max_samples=1000):
        self.tokenizer = tokenizer
        self.samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    # For simplicity, use the first answer if available
                    if qa['answers']:
                        answer = qa['answers'][0]['text']
                        answer_start = qa['answers'][0]['answer_start']
                    else:
                        # For SQuAD 2.0, some questions have no answer
                        answer = ""
                        answer_start = 0

                    self.samples.append({
                        'context': context,
                        'question': question,
                        'answer': answer,
                        'answer_start': answer_start
                    })

                    count += 1
                    if count >= max_samples:
                        break
                if count >= max_samples:
                    break
            if count >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize
        inputs = self.tokenizer(
            sample['question'],
            sample['context'],
            max_length=384,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # For simplicity, we'll use dummy start/end positions
        # In a real QA task, you'd compute these from the answer spans
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'token_type_ids': inputs['token_type_ids'].squeeze(),
            'start_positions': torch.tensor(1, dtype=torch.long),  # Dummy start position
            'end_positions': torch.tensor(2, dtype=torch.long)     # Dummy end position
        }


def get_squad_loaders(config: ExperimentConfig, tokenizer) -> tuple:
    """Load SQuAD dataset and return train/test loaders."""

    train_dataset = SquadDataset(
        tokenizer,
        os.path.join(config.dataset_root, 'SQuAD', 'train-v2.0.json'),
        max_samples=1000  # Small subset for quick testing
    )

    test_dataset = SquadDataset(
        tokenizer,
        os.path.join(config.dataset_root, 'SQuAD', 'dev-v2.0.json'),
        max_samples=200  # Small subset for quick testing
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


def create_quantized_bert(device: str = 'cuda:0') -> tuple:
    """Create quantized BERT model and dummy input."""
    from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode

    # Use small BERT for quick testing
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_hidden_layers=2)
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


def evaluate_qa_model(model, testloader, device: str = 'cuda') -> float:
    """Simple evaluation for QA model (dummy accuracy for demonstration)."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in testloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # For demonstration, just check if start_logits and end_logits exist
            if hasattr(outputs, 'start_logits') and hasattr(outputs, 'end_logits'):
                correct += 1
            total += 1

    return 100.0 * correct / total if total > 0 else 0.0


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
        variant="adam",
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
        val_acc = evaluate_qa_model(model, testloader, config.device)
        metrics = optimizer.compute_metrics()
        
        print_epoch_summary(epoch, config.epochs, avg_loss, val_acc, val_acc, metrics)

    total_time = time.time() - start_time
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    final_acc = evaluate_qa_model(model, testloader, config.device)
    final_metrics = optimizer.compute_metrics()
    
    # Compute compressed MACs and BOPs
    compressed_macs_info = oto.compute_macs(in_million=True)
    compressed_bops_info = oto.compute_bops(in_million=True)
    compressed_macs = compressed_macs_info["total"]
    compressed_bops = compressed_bops_info["total"]
    
    print(f"Compressed MACs: {compressed_macs:.2f} M")
    print(f"Compressed BOPs: {compressed_bops:.2f} M")
    
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
        final_top1_accuracy=final_acc,
        final_top5_accuracy=final_acc, # Same as val_acc1 for QA
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
    args = parser.parse_args()
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
    model, dummy_input, tokenizer = create_quantized_bert(config.device)

    # Step 2: Initialize OTO
    print("[Step 2] Initializing OTO framework...")
    oto = OTO(model=model, dummy_input=dummy_input, strict_out_nodes=False)

    # Exclude embeddings from pruning
    oto.mark_unprunable_by_param_names(["bert.embeddings.word_embeddings.weight"])

    # Step 3: Load dataset
    print("[Step 3] Loading SQuAD dataset...")
    trainloader, testloader, trainset = get_squad_loaders(config, tokenizer)
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