# ===============================================================
# Stand-alone inference script for trajectory_surrogate.pt
#
# Loads:
#   - trained neural network model
#   - normalization parameters
#   - takes an input vector (7 features)
# Produces:
#   - predicted (200 × 3) trajectory
#   - optional plots
#
# Author: Mitchell R. Stolk
# ===============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------
# MODEL DEFINITION (must match training)
# ---------------------------------------
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
        for t in range(self.seq_length):
            lstm_out, hidden = self.lstm(decoder_input, hidden)
            out = self.output_layer(lstm_out)
            outputs.append(out)
            decoder_input = out

        return torch.cat(outputs, dim=1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# -----------------------------------------------
# LOAD TRAINED MODEL + NORMALIZATION METADATA
# -----------------------------------------------
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


# -----------------------------------------------
# NORMALIZATION ARRAYS
# -----------------------------------------------
input_mean = norm["input_mean"]
input_std  = norm["input_std"]
traj_mean  = norm["traj_mean"]
traj_std   = norm["traj_std"]


# -----------------------------------------------
# EXAMPLE INPUT
# (x_target, z_target, range, bearing_deg, wx, wz, wind_speed)
# -----------------------------------------------
x_tgt = 85.0
z_tgt = 10.0

example_input = np.array([
    x_tgt,
    z_tgt,
    np.sqrt(x_tgt**2 + z_tgt**2),
    np.degrees(np.arctan2(z_tgt, x_tgt)),
    1.0,   # wx
    0.5,   # wz
    np.sqrt(1.0**2 + 0.5**2)
], dtype=np.float32)

# Normalize
inputs_norm = (example_input - input_mean.numpy()) / input_std.numpy()
inputs_norm = torch.tensor(inputs_norm, dtype=torch.float32).unsqueeze(0)

# -----------------------------------------------
# RUN INFERENCE
# -----------------------------------------------
pred_norm = model.predict(inputs_norm).squeeze(0).numpy()
pred = pred_norm * traj_std.numpy() + traj_mean.numpy()

# pred has shape (200, 3): columns → x, y, z
x, y, z = pred[:, 0], pred[:, 1], pred[:, 2]

print("\nPredicted final impact point:")
print(f"  x = {x[-1]:.2f} m")
print(f"  y = {y[-1]:.2f} m")
print(f"  z = {z[-1]:.2f} m")


# -----------------------------------------------
# OPTIONAL SIMPLE PLOT
# -----------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(x, y, label="Predicted trajectory")
plt.scatter([0], [100], c='green', label="Launch point")
plt.scatter([x_tgt],[z_tgt], c='red', label="Target")
plt.xlabel("X (m)")
plt.ylabel("Altitude (m)")
plt.grid(True)
plt.legend()
plt.title("NN Predicted Trajectory")
plt.show()
