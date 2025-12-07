# ===============================================================
# Stand-alone inference + analysis script for trajectory_surrogate.pt
#
# Provides:
#   - Full NN trajectory prediction (200 × 3)
#   - Final position, miss distance
#   - Estimated time of flight
#   - Optional error comparison vs ground truth trajectory
#   - 2D and 3D plots
#
# Author: Mitchell R. Stolk
# Date: December 2025
# ===============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# MODEL DEFINITION (must match training architecture)

class TrajectorySurrogate(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 seq_length, num_lstm_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )

        self.hidden_init = torch.nn.Linear(hidden_size, hidden_size * num_lstm_layers)
        self.cell_init = torch.nn.Linear(hidden_size, hidden_size * num_lstm_layers)

        self.lstm = torch.nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, output_size)
        )

        self.start_token = torch.nn.Parameter(torch.zeros(1, 1, output_size))

    def forward(self, x):
        batch = x.shape[0]
        encoded = self.encoder(x)

        h0 = self.hidden_init(encoded).view(batch, self.num_lstm_layers, self.hidden_size)
        c0 = self.cell_init(encoded).view(batch, self.num_lstm_layers, self.hidden_size)
        h0 = h0.permute(1, 0, 2).contiguous()
        c0 = c0.permute(1, 0, 2).contiguous()

        hidden = (h0, c0)
        decoder_input = self.start_token.expand(batch, 1, self.output_size)

        outputs = []
        for _ in range(self.seq_length):
            lstm_out, hidden = self.lstm(decoder_input, hidden)
            out = self.output_layer(lstm_out)
            outputs.append(out)
            decoder_input = out

        return torch.cat(outputs, dim=1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

# LOAD TRAINED MODEL + NORMALIZATION
MODEL_FILE = "trajectory_surrogate.pt"
checkpoint = torch.load(MODEL_FILE, map_location="cpu")

config = checkpoint["config"]
norm = checkpoint["normalization"]

model = TrajectorySurrogate(
    input_size=7,
    hidden_size=config["hidden_size"],
    output_size=len(config["predict_states"]),
    seq_length=config["seq_length"],
    num_lstm_layers=config["num_lstm_layers"],
    dropout=0.1,
)
model.load_state_dict(checkpoint["model_state_dict"])

# Normalization vectors
input_mean = norm["input_mean"].numpy()
input_std  = norm["input_std"].numpy()
traj_mean  = norm["traj_mean"].numpy()
traj_std   = norm["traj_std"].numpy()

# MAIN SURROGATE INFERENCE FUNCTION
def run_surrogate(x_tgt, z_tgt, wx, wz, verbose=True):
    """
    Returns:
        trajectory (200×3)
        final_x, final_y, final_z
        miss_distance
        est_tof
    """

    # Build input vector
    rng = np.sqrt(x_tgt**2 + z_tgt**2)
    bearing = np.degrees(np.arctan2(z_tgt, x_tgt))
    wind_speed = np.sqrt(wx**2 + wz**2)

    inp = np.array([
        x_tgt, z_tgt, rng, bearing, wx, wz, wind_speed
    ], dtype=np.float32)

    # Normalize → tensor
    inp_norm = (inp - input_mean) / input_std
    inp_norm = torch.tensor(inp_norm, dtype=torch.float32).unsqueeze(0)

    # Predict trajectory
    pred_norm = model.predict(inp_norm).squeeze(0).numpy()
    pred = pred_norm * traj_std + traj_mean

    # Extract
    x, y, z = pred[:, 0], pred[:, 1], pred[:, 2]

    # Impact time (first crossing of y ≤ 0)
    idx_ground = np.where(y <= 0.0)[0]
    est_tof = idx_ground[0] * 0.05 if len(idx_ground) > 0 else config["seq_length"] * 0.05

    miss = np.sqrt((x[-1] - x_tgt)**2 + (z[-1] - z_tgt)**2)

    if verbose:
        print("\n============================")
        print(" SURROGATE PREDICTION RESULT ")
        print("============================")
        print(f"Final X: {x[-1]:.2f} m")
        print(f"Final Y: {y[-1]:.2f} m")
        print(f"Final Z: {z[-1]:.2f} m")
        print(f"Miss Distance: {miss:.2f} m")
        print(f"Estimated TOF: {est_tof:.2f} s")
        print("----------------------------")

    return pred, (x[-1], y[-1], z[-1]), miss, est_tof

# OPTIONAL PLOTTING
def plot_all(pred, x_tgt, z_tgt):
    x, y, z = pred[:, 0], pred[:, 1], pred[:, 2]

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, label="NN Trajectory")
    plt.scatter([0], [100], c='green', label="Launch")
    plt.scatter([x_tgt], [0], c='red', label="Target")
    plt.xlabel("X (m)")
    plt.ylabel("Altitude (m)")
    plt.grid()
    plt.legend()
    plt.title("Trajectory (X-Y)")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(x, z, label="Top-Down")
    plt.scatter([x_tgt], [z_tgt], c='red', label="Target")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.legend()
    plt.grid()
    plt.title("Top-Down (X-Z)")
    plt.show()

    # 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, z, y, label="NN Trajectory")
    ax.scatter([x_tgt], [z_tgt], [0], c="red", s=80)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title("3D Trajectory")
    plt.show()
# EXAMPLE RUN
if __name__ == "__main__":
    pred, final_pos, miss, tof = run_surrogate(
        x_tgt=85.0,
        z_tgt=10.0,
        wx=1.0,
        wz=0.5,
        verbose=True
    )
def compare_to_ground_truth(index, dataset_path="trajectories.npz"):
    """
    Loads one ground-truth trajectory from trajectories.npz
    and compares surrogate vs truth on the same plot.
    """

    print(f"\nComparing surrogate prediction to ground truth for index {index}")

    data = np.load(dataset_path)

    # Ground truth arrays
    inputs = data["inputs"]        # (N, 7)
    trajs  = data["trajectories"]  # (N, 200, 13)
    times  = data["times"]         # (200,)

    # Extract sample
    inp = inputs[index]
    truth = trajs[index, :, :]     # (200, 13)
    truth_xyz = truth[:, :3]       # x, y, z only

    # --- Run surrogate ---
    x_tgt = inp[0]
    z_tgt = inp[1]
    wx    = inp[4]
    wz    = inp[5]

    pred, _, _, _ = run_surrogate(x_tgt, z_tgt, wx, wz, verbose=False)
    pred_xyz = pred[:, :3]

    # PLOTS
    # X-Y altitude plot
    plt.figure(figsize=(10,5))
    plt.plot(truth_xyz[:,0], truth_xyz[:,1], 'b-', label="Ground Truth")
    plt.plot(pred_xyz[:,0],  pred_xyz[:,1],  'r--', label="Surrogate")
    plt.xlabel("X (m)")
    plt.ylabel("Altitude Y (m)")
    plt.title(f"Trajectory Comparison (Index {index})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Top-down X-Z plot
    plt.figure(figsize=(10,5))
    plt.plot(truth_xyz[:,0], truth_xyz[:,2], 'b-', label="Ground Truth")
    plt.plot(pred_xyz[:,0],  pred_xyz[:,2],  'r--', label="Surrogate")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(f"Top-Down Comparison (Index {index})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3D plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth_xyz[:,0], truth_xyz[:,2], truth_xyz[:,1], 'b-', label="Ground Truth")
    ax.plot(pred_xyz[:,0],  pred_xyz[:,2],  pred_xyz[:,1],  'r--', label="Surrogate")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title(f"3D Trajectory Comparison (Index {index})")
    ax.legend()
    plt.show()

if __name__ == "__main__":

    # 1) Standalone surrogate prediction
    pred, final_pos, miss, tof = run_surrogate(
        x_tgt=85.0,
        z_tgt=10.0,
        wx=1.0,
        wz=0.5,
        verbose=True
    )

    # 3) Plot surrogate-only trajectory (correct target coords)
    plot_all(pred, 85.0, 10.0)

    # Comparison vs ground truth trajectory from npz dataset
    compare_to_ground_truth(index=123)  # pick any sample index


    plot_all(pred, 85.0, 10.0)
