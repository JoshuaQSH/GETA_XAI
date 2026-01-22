import sys
from tqdm import tqdm
import numpy as np

# sys.path.append('..')
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from sanity_check.backends.vgg7 import vgg7_bn
from only_train_once import OTO
import torch

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# This follows the GETA github tutorial for VGG7 on CIFAR-10
# https://github.com/microsoft/geta/blob/0518a57021eeed0b6f74eab709730dcf5b0135f1/tutorials/03.qvgg7_cifar10.ipynb

dataset_root = '../datasets'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Step 0. Helpers
def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_accuracy(model, testloader, two_input=False):
    correct1 = 0
    correct5 = 0
    total = 0
    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, dict):  # ImageNet format
                X = batch['pixel_values'].to(device)
                y = batch['labels'].to(device)
            else:  # CIFAR10 format
                X, y = batch
                X = X.to(device)
                y = y.to(device)
            
            if two_input:
                y_pred = model(X, X)
            else:
                y_pred = model(X)
            
            total += y.size(0)

            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))
            
            correct1 += prec1.item() * y.size(0)
            correct5 += prec5.item() * y.size(0)

    model = model.train()
    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5

# Step 1. Create OTO instance
model = vgg7_bn()
model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

# Step 2. Dataset Preparation
trainset = CIFAR10(root=dataset_root, train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
testset = CIFAR10(root=dataset_root, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

trainloader =  torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

# Step 3. Setup the optimizer
"""
variant: The optimizer that is used for training the baseline full model. Currently support sgd, adam and adamw.
lr: The initial learning rate.
weight_decay: Weight decay as standard DNN optimization.
target_group_sparsity: The target group sparsity, typically higher group sparsity refers to more FLOPs and model size reduction, meanwhile may regress model performance more.
start_projection_step: The number of steps that starts to do bit width projection.
projection_steps: The number of steps that finishes bit width projection (reach target bit width) after start_projection_steps.
projection_periods: Incrementally produce the group sparsity equally among projection periods.
start_pruning_step: The number of steps that starts to prune.
pruning_steps: The number of steps that finishes pruning (reach target_group_sparsity) after start_pruning_steps.
pruning_periods: Incrementally produce the group sparsity equally among pruning periods.
bit reduction: the reduction of max_bit after the end of each projection period.
[min_bit,max_bit]: Initial bit width interval. The max_bit will reduce by bit_reduction after each projection period.
"""

def train_geta(trainloader, testloader, model, oto, max_epoch=50, device=DEVICE):
    optimizer = oto.geta(
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        start_projection_step=0 * len(trainloader),
        projection_periods=5,
        projection_steps=10 * len(trainloader),
        start_pruning_step=10 * len(trainloader),
        pruning_periods=5,
        pruning_steps=10 * len(trainloader),
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
    )

    # Step 4. Train VGG7 as normal.
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # Every 50 epochs, decay lr by 10.0
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) 

    for epoch in range(max_epoch):
        f_avg_val = 0.0
        model.train()
        lr_scheduler.step()
        for X, y in trainloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model.forward(X)
            f = criterion(y_pred, y)
            optimizer.zero_grad()
            f.backward()
            f_avg_val += f
            optimizer.step()
        opt_metrics = optimizer.compute_metrics()
        accuracy1, accuracy5 = check_accuracy(model, testloader)
        f_avg_val = f_avg_val.cpu().item() / len(trainloader)
        
        print("Ep: {ep}, loss: {f:.2f}, norm_all:{param_norm:.2f}, grp_sparsity: {gs:.2f}, acc1: {acc1:.4f}, norm_import: {norm_import:.2f}, norm_redund: {norm_redund:.2f}, num_grp_import: {num_grps_import}, num_grp_redund: {num_grps_redund}"\
            .format(ep=epoch, f=f_avg_val, param_norm=opt_metrics.norm_params, gs=opt_metrics.group_sparsity, acc1=accuracy1,\
            norm_import=opt_metrics.norm_important_groups, norm_redund=opt_metrics.norm_redundant_groups, \
            num_grps_import=opt_metrics.num_important_groups, num_grps_redund=opt_metrics.num_redundant_groups
            ))
    
    return model, oto.compressed_model_path

def get_compressed_model_size(compressed_model_path, device=DEVICE):
    # By default OTO will construct subnet by the last checkpoint. If intermedia ckpt reaches the best performance,
    # need to reinitialize OTO instance
    # oto = OTO(torch.load(ckpt_path), dummy_input)
    # then construct subnetwork
    dummy_input = torch.rand(1, 3, 32, 32)
    oto.construct_subnet(out_dir='./cache')
    compressed_model = torch.load(compressed_model_path)
    oto_compressed = OTO(compressed_model, dummy_input.to(device))
    
    # Get full model MACs, BOPs
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_num_params = oto.compute_num_params(in_million=True)

    # Get compressed model MACs, BOPs
    compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)
    compressed_bops = oto_compressed.compute_bops(in_million=True, layerwise=True)

    print(f"Full MACs for VGG7: {full_macs['total']} M MACs")
    print(f"Full BOPs for VGG7: {full_bops['total']} M BOPs")
    print(f"Compressed MACs for VGG7: {compressed_macs['total']} M MACs")
    print(f"Compressed BOPs for VGG7: {compressed_bops['total']} M BOPs")

compressed_model, compressed_model_path = train_geta(trainloader, testloader, model, oto, max_epoch=50, device=DEVICE)
get_compressed_model_size(compressed_model_path, device=DEVICE)