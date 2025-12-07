# Trajectory Surrogate Neural Network Training
# Learns to predict full 6-DoF trajectories from initial conditions
#
# Author: Mitchell R. Stolk
# License: MIT
# Date: December 2025
#
# Input: (x_target, z_target, range, bearing, wx, wz, wind_speed) - 7 features
# Output: (x, y, z) trajectory over 200 timesteps - (200, 3)
#
# Architecture: MLP Encoder → LSTM Decoder
#
# Usage:
#   python train_trajectory_surrogate.py
#
# Requirements:
#   pip install torch numpy matplotlib tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

# === CONFIGURATION ===
DATA_FILE = "trajectories.npz"
MODEL_FILE = "trajectory_surrogate.pt"

# Training settings
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 100
TRAIN_SPLIT = 0.9  # 90% train, 10% validation

# Model architecture
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.1

# Which states to predict (indices into 13-state vector)
# 0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz, 6-9=quaternion, 10-12=angular rates
PREDICT_STATES = [0, 1, 2]  # Just position (x, y, z) to start
STATE_NAMES = ['x', 'y', 'z']

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrajectoryDataset(Dataset):
    """Dataset for trajectory prediction."""
    
    def __init__(self, inputs, trajectories, tofs, predict_indices):
        """
        Args:
            inputs: (N, 7) array of input features
            trajectories: (N, T, 13) array of full state trajectories
            tofs: (N,) array of times of flight
            predict_indices: list of state indices to predict
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        # Extract only the states we want to predict
        self.trajectories = torch.tensor(
            trajectories[:, :, predict_indices], dtype=torch.float32
        )
        self.tofs = torch.tensor(tofs, dtype=torch.float32)
        
        # Normalize inputs (store mean/std for inference)
        self.input_mean = self.inputs.mean(dim=0)
        self.input_std = self.inputs.std(dim=0) + 1e-8
        self.inputs_normalized = (self.inputs - self.input_mean) / self.input_std
        
        # Normalize trajectories per state
        self.traj_mean = self.trajectories.mean(dim=(0, 1))
        self.traj_std = self.trajectories.std(dim=(0, 1)) + 1e-8
        self.trajectories_normalized = (self.trajectories - self.traj_mean) / self.traj_std
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return (
            self.inputs_normalized[idx],
            self.trajectories_normalized[idx],
            self.tofs[idx]
        )
    
    def get_normalization_params(self):
        """Return normalization parameters for saving with model."""
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'traj_mean': self.traj_mean,
            'traj_std': self.traj_std
        }


class TrajectorySurrogate(nn.Module):
    """
    Neural network that predicts full trajectories from input conditions.
    
    Architecture:
        Input (7) → MLP Encoder → Hidden State → LSTM Decoder → Trajectory (T, 3)
    """
    
    def __init__(self, input_size=7, hidden_size=256, output_size=3, 
                 seq_length=200, num_lstm_layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers
        
        # Encoder: Input features → hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Initialize LSTM hidden states from encoded input
        self.hidden_init = nn.Linear(hidden_size, hidden_size * num_lstm_layers)
        self.cell_init = nn.Linear(hidden_size, hidden_size * num_lstm_layers)
        
        # LSTM decoder: generates sequence
        self.lstm = nn.LSTM(
            input_size=output_size,  # Previous output as input
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Learnable start token
        self.start_token = nn.Parameter(torch.zeros(1, 1, output_size))
        
    def forward(self, x, teacher_forcing_ratio=0.0, target=None):
        """
        Forward pass.
        
        Args:
            x: (batch, 7) input features
            teacher_forcing_ratio: probability of using ground truth as next input
            target: (batch, seq_length, output_size) ground truth for teacher forcing
            
        Returns:
            outputs: (batch, seq_length, output_size) predicted trajectory
        """
        batch_size = x.shape[0]
        
        # Encode input
        encoded = self.encoder(x)  # (batch, hidden_size)
        
        # Initialize LSTM hidden states
        h0 = self.hidden_init(encoded)  # (batch, hidden_size * num_layers)
        c0 = self.cell_init(encoded)
        
        # Reshape for LSTM: (num_layers, batch, hidden_size)
        h0 = h0.view(batch_size, self.num_lstm_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = c0.view(batch_size, self.num_lstm_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        
        # Generate sequence autoregressively
        outputs = []
        
        # Start with learnable start token
        decoder_input = self.start_token.expand(batch_size, 1, -1)
        hidden = (h0, c0)
        
        for t in range(self.seq_length):
            # LSTM step
            lstm_out, hidden = self.lstm(decoder_input, hidden)
            
            # Predict output
            output = self.output_layer(lstm_out)  # (batch, 1, output_size)
            outputs.append(output)
            
            # Next input: teacher forcing or own prediction
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = output
        
        # Stack outputs: (batch, seq_length, output_size)
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def predict_fast(self, x):
        """Fast inference without teacher forcing."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, teacher_forcing_ratio=0.0)


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio, scaler):
    model.train()
    total_loss = 0

    for inputs, targets, tofs in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # AMP forward pass
        with torch.cuda.amp.autocast():
            outputs = model(inputs, teacher_forcing_ratio=teacher_forcing_ratio, target=targets)
            loss = criterion(outputs, targets)

        # AMP backward + optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)




def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets, tofs in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs, teacher_forcing_ratio=0.0)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_trajectory_comparison(model, dataset, device, num_examples=6, save_path='trajectory_comparison.png'):
    """Plot predicted vs actual trajectories."""
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Random indices
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            inputs, targets, tof = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)
            
            # Predict
            pred = model.predict_fast(inputs).cpu().squeeze()
            
            # Denormalize
            pred = pred * dataset.traj_std + dataset.traj_mean
            targets = targets * dataset.traj_std + dataset.traj_mean
            
            # Plot X-Y trajectory (top-down would be X-Z, but let's do X-Y for altitude)
            ax = axes[i]
            ax.plot(targets[:, 0], targets[:, 1], 'b-', linewidth=2, label='Ground Truth')
            ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='Predicted')
            ax.scatter(targets[0, 0], targets[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
            ax.scatter(targets[-1, 0], targets[-1, 1], c='blue', s=100, marker='x', zorder=5)
            ax.scatter(pred[-1, 0], pred[-1, 1], c='red', s=100, marker='x', zorder=5)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y - Altitude (m)')
            ax.set_title(f'Trajectory {idx}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Predicted vs Ground Truth Trajectories (X-Y View)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_3d_comparison(model, dataset, device, num_examples=4, save_path='trajectory_3d_comparison.png'):
    """Plot 3D trajectory comparisons."""
    model.eval()
    
    fig = plt.figure(figsize=(16, 12))
    
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            inputs, targets, tof = dataset[idx]
            inputs_tensor = inputs.unsqueeze(0).to(device)
            
            pred = model.predict_fast(inputs_tensor).cpu().squeeze()
            
            # Denormalize
            pred = pred * dataset.traj_std + dataset.traj_mean
            targets = targets * dataset.traj_std + dataset.traj_mean
            
            # Denormalize inputs to get target location
            inp_denorm = inputs * dataset.input_std + dataset.input_mean
            x_tgt, z_tgt = inp_denorm[0].item(), inp_denorm[1].item()
            
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            
            # Plot trajectories
            ax.plot(targets[:, 0].numpy(), targets[:, 2].numpy(), targets[:, 1].numpy(), 
                   'b-', linewidth=2, label='Ground Truth')
            ax.plot(pred[:, 0].numpy(), pred[:, 2].numpy(), pred[:, 1].numpy(), 
                   'r--', linewidth=2, label='Predicted')
            
            # Start point
            ax.scatter(targets[0, 0], targets[0, 2], targets[0, 1], 
                      c='green', s=100, marker='o', label='Launch')
            
            # Target
            ax.scatter(x_tgt, z_tgt, 0, c='purple', s=150, marker='*', label='Target')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.set_zlabel('Altitude (m)')
            ax.set_title(f'Trajectory {idx}')
            ax.legend(fontsize=8)
    
    plt.suptitle('3D Trajectory Comparison: Predicted vs Ground Truth', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def compute_metrics(model, dataset, device):
    """Compute accuracy metrics on the dataset."""
    model.eval()
    
    all_errors = []
    position_errors_final = []
    
    with torch.no_grad():
        # Process in batches
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        for inputs, targets, tofs in tqdm(loader, desc='Computing metrics'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            pred = model.predict_fast(inputs)
            
            # Denormalize
            pred = pred * dataset.traj_std.to(device) + dataset.traj_mean.to(device)
            targets = targets * dataset.traj_std.to(device) + dataset.traj_mean.to(device)
            
            # Position error at each timestep
            errors = torch.sqrt(((pred - targets) ** 2).sum(dim=-1))  # (batch, seq_len)
            all_errors.append(errors.cpu())
            
            # Final position error
            final_error = errors[:, -1]
            position_errors_final.append(final_error.cpu())
    
    all_errors = torch.cat(all_errors, dim=0)
    position_errors_final = torch.cat(position_errors_final, dim=0)
    
    metrics = {
        'mean_trajectory_error': all_errors.mean().item(),
        'max_trajectory_error': all_errors.max().item(),
        'mean_final_position_error': position_errors_final.mean().item(),
        'std_final_position_error': position_errors_final.std().item(),
        'median_final_position_error': position_errors_final.median().item(),
        '95th_percentile_error': torch.quantile(position_errors_final, 0.95).item(),
    }
    
    return metrics


def main():
    print("="*60)
    print("TRAJECTORY SURROGATE NEURAL NETWORK TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Data file: {DATA_FILE}")
    print(f"Predicting states: {STATE_NAMES}")
    print("="*60)
    
    # Load data
    print("\nLoading trajectory data...")
    data = np.load(DATA_FILE)
    inputs = data['inputs']
    trajectories = data['trajectories']
    tofs = data['tofs']
    times = data['times']
    
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  TOFs shape: {tofs.shape}")
    print(f"  Time steps: {len(times)}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = TrajectoryDataset(inputs, trajectories, tofs, PREDICT_STATES)
    
    # Split into train/val
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"  Training samples: {train_size:,}")
    print(f"  Validation samples: {val_size:,}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Create model
    print("\nCreating model...")
    model = TrajectorySurrogate(
        input_size=7,
        hidden_size=HIDDEN_SIZE,
        output_size=len(PREDICT_STATES),
        seq_length=trajectories.shape[1],
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Teacher forcing schedule: start high, decay to 0
    teacher_forcing_start = 0.5
    teacher_forcing_end = 0.0
    
    t_start = time.perf_counter()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(EPOCHS):
        # Compute teacher forcing ratio (linear decay)
        teacher_forcing_ratio = teacher_forcing_start - (teacher_forcing_start - teacher_forcing_end) * (epoch / EPOCHS)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, teacher_forcing_ratio, scaler)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'normalization': dataset.get_normalization_params(),
                'config': {
                    'hidden_size': HIDDEN_SIZE,
                    'num_lstm_layers': NUM_LSTM_LAYERS,
                    'seq_length': trajectories.shape[1],
                    'predict_states': PREDICT_STATES,
                    'state_names': STATE_NAMES,
                }
            }, MODEL_FILE)
        
        # Progress
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"TF: {teacher_forcing_ratio:.2f} | "
              f"Best: {best_val_loss:.6f} | "
              f"ETA: {eta/60:.1f}min")
    
    total_time = time.perf_counter() - t_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(MODEL_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training curves
    print("\nGenerating training curves plot...")
    plot_training_curves(train_losses, val_losses)
    
    # Plot trajectory comparisons
    print("\nGenerating trajectory comparison plots...")
    plot_trajectory_comparison(model, dataset, DEVICE)
    plot_3d_comparison(model, dataset, DEVICE)
    
    # Compute final metrics
    print("\nComputing final metrics...")
    metrics = compute_metrics(model, dataset, DEVICE)
    
    print("\n" + "="*60)
    print("FINAL METRICS")
    print("="*60)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f} m")
    
    # Speed comparison
    print("\n" + "="*60)
    print("SPEED COMPARISON")
    print("="*60)
    
    # Time neural network inference
    model.eval()
    test_input = torch.randn(1000, 7).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        _ = model.predict_fast(test_input)
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model.predict_fast(test_input)
    torch.cuda.synchronize()
    nn_time = (time.perf_counter() - t0) / 10
    
    nn_rate = 1000 / nn_time
    physics_rate = 30  # Your approximate physics sim rate
    
    print(f"  Neural network: {nn_rate:,.0f} trajectories/sec")
    print(f"  Physics sim: ~{physics_rate} trajectories/sec")
    print(f"  Speedup: {nn_rate/physics_rate:.0f}x faster!")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Model size: {os.path.getsize(MODEL_FILE)/(1024*1024):.1f} MB")
    print("\nTo use the model:")
    print("  from train_trajectory_surrogate import TrajectorySurrogate")
    print("  model = TrajectorySurrogate(...)")
    print("  model.load_state_dict(torch.load('trajectory_surrogate.pt')['model_state_dict'])")
    print("  trajectory = model.predict_fast(inputs)")


if __name__ == '__main__':
    main()