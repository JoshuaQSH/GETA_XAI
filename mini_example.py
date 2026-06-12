from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from sanity_check.backends.vgg7 import vgg7_bn
from only_train_once import OTO
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import os

def get_trainloaders(dataset_root):
    """Load CIFAR-10 dataset and return train/test loaders."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = CIFAR10(
        root=dataset_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    trainloader = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return trainloader 


# Create OTO instance
model = vgg7_bn()
model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

# CIFAR10 dataset loading
dataset_root = os.environ.get('MY_SCRATCH') + '/datasets'
trainloader = get_trainloaders(dataset_root)

# Create GETA optimizer
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

# Train the DNN as normal via GETA
model.train()
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
max_epoch = 10
for epoch in range(max_epoch):
    f_avg_val = 0.0
    for X, y in trainloader:
        X, y = X.cuda(), y.cuda()
        y_pred = model.forward(X)
        f = criterion(y_pred, y)
        optimizer.zero_grad()
        f.backward()
        optimizer.step()

# A pruned and quantized vgg7 will be generated. 
oto.construct_subnet(out_dir='./')
