#!/usr/bin/env python3
"""
XAI-GETA vs GETA Comparison Plots

This script generates:
1. Training loss comparison (line plot)
2. Final accuracy comparison (bar plot)
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import os
import argparse

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

color_style = ['#ECDFFF', '#D4C4EE', '#BDAADE', '#A690CF', '#8E78BF', '#8C75BC', '#6A579C', '#aed0ee', '#88abda', '#6f94cd', '#5976ba', '#2e59a7', '#145ca0']

# Multi xai logs example:
LOG_FILE_LIST = ['/scratch/staff/lrr550/geta/logs/input_x_gradient_0.3_50epoch.log',
                 '/scratch/staff/lrr550/geta/logs/deconvolution_0.3_50epoch.log',
                 '/scratch/staff/lrr550/geta/logs/guided_backprop_0.3_50epoch.log',
                #  '/scratch/staff/lrr550/geta/logs/layer_conductance_0.3_50epoch.log',
                 '/scratch/staff/lrr550/geta/logs/saliency_0.3_50epoch.log',
                 '/scratch/staff/lrr550/geta/logs/deconvolution_0.3_50epoch.log',
                 '/scratch/staff/lrr550/geta/logs/deep_lift_0.3_50epoch.log',
                 '/scratch/staff/lrr550/geta/logs/integrated_gradients_0.3_50epoch.log',
                 '/scratch/staff/lrr550/geta/logs/layer_lrp_0.3_50epoch.log',
                 '/scratch/staff/lrr550/geta/logs/lrp_0.3_50epoch.log', ]

# LOG_FILE_LIST = ['/scratch/staff/lrr550/geta/logs/input_x_gradient_0.5_50epoch.log',
#                  '/scratch/staff/lrr550/geta/logs/deconvolution_0.5_50epoch.log',
#                  '/scratch/staff/lrr550/geta/logs/guided_backprop_0.5_50epoch.log',
#                 #  '/scratch/staff/lrr550/geta/logs/layer_conductance_0.5_50epoch.log',
#                 #  '/scratch/staff/lrr550/geta/logs/saliency_0.5_50epoch.log',
#                  '/scratch/staff/lrr550/geta/logs/deconvolution_0.5_50epoch.log',
#                  '/scratch/staff/lrr550/geta/logs/deep_lift_0.5_50epoch.log',
#                  '/scratch/staff/lrr550/geta/logs/integrated_gradients_0.5_50epoch.log',
#                  '/scratch/staff/lrr550/geta/logs/layer_lrp_0.5_50epoch.log',
#                  '/scratch/staff/lrr550/geta/logs/lrp_0.5_50epoch.log', ]


def parse_arguments():
    parser = argparse.ArgumentParser(
            prog='Plotting script for XAI-GETA vs GETA comparison',
            description='Generates plots comparing training loss and final accuracy between XAI-GETA and original GETA.')
    
    parser.add_argument('--xai-path', type=str, default='xai_geta_50epoch_results.log',
                        required=False, help='Path to XAI-GETA log file.')
    parser.add_argument('--multi-logs', action='store_true', 
                        required=False, default=False, help='Indicates if the XAI-GETA log file contains multiple runs/logs')
    parser.add_argument('--geta-path', type=str, default='geta_50epoch.log',
                        required=False, help='Path to original GETA log file.')
    parser.add_argument('--config-path', type=str, default='./run_cases/configs/default_xai_geta.yaml', 
                        required=False, help='Path to experiment config YAML file.')
    parser.add_argument('--output-loss', type=str, default='comparison_training_loss.png',
                        required=False, help='Path to output training loss plot.')
    parser.add_argument('--output-accuracy', type=str, default='comparison_final_accuracy.png',
                        required=False, help='Path to output final accuracy plot.')
    
    args = parser.parse_args()
    return args

def parse_xai_log(log_path):
    """Parse XAI-GETA log file to extract metrics."""
    epochs = []
    losses = []
    accuracies = []
    
    # Pattern: Ep:  1 | Loss: 1.7945 | Acc@1: 39.24%
    pattern = r"Ep:\s*(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Acc@1:\s*([\d.]+)%"
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accuracies.append(float(match.group(3)))
    
    return epochs, losses, accuracies


def parse_geta_log(log_path):
    """Parse original GETA log file to extract metrics."""
    epochs = []
    losses = []
    accuracies = []
    
    # Pattern: Ep: 0, loss: 1.75, norm_all:4542.88, grp_sparsity: 0.00, acc1: 42.6200
    pattern = r"Ep:\s*(\d+),\s*loss:\s*([\d.]+),.*acc1:\s*([\d.]+)"
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)) + 1)  # Convert to 1-indexed
                losses.append(float(match.group(2)))
                accuracies.append(float(match.group(3)))
    
    return epochs, losses, accuracies


def plot_training_loss(xai_data, geta_data, output_path, configs):
    """Create training loss comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    xai_epochs, xai_losses, _ = xai_data
    geta_epochs, geta_losses, _ = geta_data
        
    # Plot both loss curves
    ax.plot(xai_epochs, xai_losses, 'b-', linewidth=2.5, marker='o', 
            markersize=4, label='XAI-GETA', alpha=0.9)
    ax.plot(geta_epochs, geta_losses, 'r-', linewidth=2.5, marker='s', 
            markersize=4, label='GETA (Original)', alpha=0.9)
    
    # Warmup Start epoch: None (default 0), End: start_projection_step
    # Projection Start epoch: start_projection_step, End epoch: start_projection_step + projection_epochs
    # Pruning Start epoch: start_projection_step + projection_epochs, End epoch: start_projection_step + projection_epochs + pruning_epochs
    # Fine-tuning Start epoch: start_projection_step + projection_epochs + pruning_epochs, End epoch: epochs
    
    # Add vertical line at Projection start
    ax.axvline(x=configs.start_projection_step, color='#6A579C', linestyle='--', alpha=0.7, label='Projection Starts')
    
    # Add vertical line at end of Pruning
    ax.axvline(x=configs.start_projection_step + configs.projection_epochs, color='#6A579C', linestyle=':', alpha=0.7, label='Pruning Starts')
    
    # Add vertical line at end of Pruning
    ax.axvline(x=configs.start_projection_step + configs.projection_epochs + configs.pruning_epochs, color='#6A579C', linestyle='-.', alpha=0.7, label='Fine-tuning Starts')
    
    # Customize plot
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss Comparison: XAI-GETA vs GETA\n(VGG7 on CIFAR-10, 50 Epochs)', 
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim([0, len(xai_epochs)+1])
    ax.set_ylim([0, 2.0])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add annotation for key observations
    # ax.annotate('XAI-GETA\nFaster Convergence', xy=(35, 0.35), fontsize=10, 
    #             ha='center', color='blue', alpha=0.8)
    # ax.annotate('GETA\nSlower Recovery', xy=(35, 0.60), fontsize=10, 
    #             ha='center', color='red', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_accuracy(xai_data, geta_data, output_path):
    """Create final accuracy bar comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    _, _, xai_acc = xai_data
    _, _, geta_acc = geta_data
    
    # Get final accuracies
    xai_final = xai_acc[-1] if xai_acc else 0
    geta_final = geta_acc[-1] if geta_acc else 0
    
    # Create bar plot
    methods = ['XAI-GETA\n(Captum Saliency)', 'GETA\n(Original)']
    accuracies = [xai_final, geta_final]
    colors = ['#D4C4EE', '#8C75BC']
    
    bars = ax.bar(methods, accuracies, color=colors, width=0.5, linewidth=0)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 8),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=16, fontweight='bold')
    
    # Add improvement annotation
    improvement = xai_final - geta_final
    ax.annotate(f'+{improvement:.2f}% Improvement',
               xy=(0.5, max(accuracies) + 3),
               ha='center', fontsize=14, color='green', fontweight='bold')
    
    # Customize plot
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax.set_title('Final Accuracy Comparison: XAI-GETA vs GETA\n(VGG7 on CIFAR-10 after 50 Epochs)', 
                 fontweight='bold', pad=15)
    ax.set_ylim([0, 100])
    
    # Add horizontal reference line at 80%
    # ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% Baseline')
    
    # Add grid (only horizontal)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend for improvement indicator
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor='#D4C4EE', label=f'XAI-GETA'),
    #     Patch(facecolor='#8C75BC', label=f'GETA')
    # ]
    # ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_summary_table(xai_data, geta_data):
    """Print summary table comparing the two methods."""
    xai_epochs, xai_losses, xai_acc = xai_data
    geta_epochs, geta_losses, geta_acc = geta_data
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY: XAI-GETA vs GETA")
    print("="*70)
    print(f"{'Metric':<30} {'XAI-GETA':<18} {'GETA':<18}")
    print("-"*70)
    print(f"{'Final Accuracy':<30} {xai_acc[-1]:.2f}%{'':<12} {geta_acc[-1]:.2f}%")
    print(f"{'Final Loss':<30} {xai_losses[-1]:.4f}{'':<12} {geta_losses[-1]:.4f}")
    print(f"{'Min Loss (all epochs)':<30} {min(xai_losses):.4f}{'':<12} {min(geta_losses):.4f}")
    print(f"{'Max Accuracy (all epochs)':<30} {max(xai_acc):.2f}%{'':<12} {max(geta_acc):.2f}%")
    print("-"*70)
    print(f"{'Accuracy Improvement':<30} +{xai_acc[-1] - geta_acc[-1]:.2f}%")
    print(f"{'Loss Reduction':<30} {(geta_losses[-1] - xai_losses[-1]) / geta_losses[-1] * 100:.1f}% lower")
    print("="*70)
    
def plot_training_loss_multi_xailogs(epochs_list, losses_list, xai_name, xai_weights, output_path):
    """Create training loss comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for epochs, losses, name, weight in zip(epochs_list, losses_list, xai_name, xai_weights):
        label = f"{name} (weight={weight})" if weight is not None else name
        ax.plot(epochs, losses, linewidth=2.5, marker='o', 
                markersize=4, label=label, alpha=0.9)
    
    # Add vertical line at pruning start (epoch 11)
    ax.axvline(x=11, color='#6A579C', linestyle='--', alpha=0.7, label='Pruning Start')
    
    # Add vertical line at end of pruning (epoch 20)
    ax.axvline(x=20, color='#6A579C', linestyle=':', alpha=0.7, label='Pruning End')
    
    # Customize plot
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss Comparison with various XAI Attributions \n(VGG7 on CIFAR-10, 50 Epochs)', 
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim([0, 51])
    ax.set_ylim([0, 2.0])
    
    # Add grid
    ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_accuracy_multi_xailogs(accuracies_list, xai_name, xai_weights, output_path):
    """Create final accuracy bar comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create bar plot
    num_methods = len(accuracies_list)
    colors = color_style[:num_methods]
    methods = []
    final_acc = [0] * num_methods
    for n, w in zip(xai_name, xai_weights):
        xai_label = f"{n} (weight={w})" if w is not None else n
        methods.append(xai_label)
    
    for i in range(len(accuracies_list)):
        final_acc[i] = accuracies_list[i][-1]  # Get final accuracy only

    bars = []
    for method, acc, color in zip(methods, final_acc, colors):
        bar = ax.bar(
            method,
            acc,
            color=color,
            width=0.8,
            linewidth=0,
            label=f"{method}"
        )
        bars.append(bar)
    
    for bar, acc in zip(bars, final_acc):
        height = bar[0].get_height()
        ax.annotate(f'{acc:.2f}%',
                   xy=(bar[0].get_x() + bar[0].get_width() / 2, height),
                   xytext=(0, 8),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=6, fontweight='bold')
    
    # Customize plot
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax.set_title('Final Accuracy with XAI Attributions\n(VGG7 on CIFAR-10 after 50 Epochs)', 
                 fontweight='bold', pad=15)
    ax.set_ylim([0, 100])
    
    # Remove x-axis tick labels
    ax.set_xticks([])
    ax.set_xlabel("")
    
    # Add horizontal reference line at 90%
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid (only horizontal)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Upright legend (vertical) placed outside plot
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def xai_geta_plotting(xai_log, geta_log, loss_plot, accuracy_plot, configs):
    # Check if log files exist
    if not os.path.exists(xai_log):
        print(f"Error: XAI-GETA log not found: {xai_log}")
        return
    if not os.path.exists(geta_log):
        print(f"Error: GETA log not found: {geta_log}")
        return
    
    print("Parsing log files...")
    
    # Parse logs
    xai_data = parse_xai_log(xai_log)
    geta_data = parse_geta_log(geta_log)
    
    print(f"XAI-GETA: {len(xai_data[0])} epochs parsed")
    print(f"GETA: {len(geta_data[0])} epochs parsed")
    
    # Create plots
    print("\nGenerating plots...")
    plot_training_loss(xai_data, geta_data, loss_plot, configs)
    plot_final_accuracy(xai_data, geta_data, accuracy_plot)
    
    # Print summary
    create_summary_table(xai_data, geta_data)
    
    print("\n All plots generated successfully!")
    print(f"   - Training Loss: {loss_plot}")
    print(f"   - Final Accuracy: {accuracy_plot}")

def plot_multiple_xai_logs(log_file_list, output_path='comparison_training_loss_multi_xailogs.png'):
    epochs_list, losses_list, accuracies_list, xai_name, xai_weights = [], [], [], [], []
    for log_path in log_file_list:
        e, l, a = parse_xai_log(log_path)
        epochs_list.append(e)
        losses_list.append(l)
        accuracies_list.append(a)
        name_temp = os.path.basename(log_path).replace('.log','')
        xai_name.append(re.search(r'^(.*)_0\.\d+_', name_temp).group(1))
        weight_match = re.search(r'_(0\.\d+)_', log_path)
        if weight_match:
            xai_weights.append(float(weight_match.group(1)))
        else:
            xai_weights.append(None)
    plot_training_loss_multi_xailogs(epochs_list, losses_list, xai_name, xai_weights, output_path)
    plot_final_accuracy_multi_xailogs(accuracies_list, xai_name, xai_weights, output_path.replace('loss','final_accuracy'))

def get_configs(config_path):
    from run_cases.utils import load_config_from_yaml, yaml_to_experiment_config
    yaml_config = load_config_from_yaml(config_path)
    config = yaml_to_experiment_config(yaml_config)
    
    return config
    
    
def main():
    # File paths
    args = parse_arguments()
    xai_log = args.xai_path
    geta_log = args.geta_path
    
    # Output paths
    loss_plot = args.output_loss
    accuracy_plot = args.output_accuracy
    
    # Read config file
    configs = get_configs(args.config_path)
    print(f"Using config from: {args.config_path}")
    
     # Plotting
    if args.multi_logs:
        plot_multiple_xai_logs(LOG_FILE_LIST, output_path='comparison_training_loss_multi_xailogs.png')
    else:
        xai_geta_plotting(xai_log, geta_log, loss_plot, accuracy_plot, configs)


if __name__ == "__main__":
    main()
