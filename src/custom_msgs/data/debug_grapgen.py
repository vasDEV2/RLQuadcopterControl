import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------
# Load CSV
# -----------------------
csv_path = "/home/control-lab/px4_ws/src/custom_msgs/dataset_debug/debug_hard_H1.csv"
df = pd.read_csv(csv_path)

# -----------------------
# Output directory
# -----------------------
out_dir = "/home/control-lab/px4_ws/src/custom_msgs/dataset_debug/plots_hard"
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# Time vector
# -----------------------
t = df["sec"] + df["nsecs"] * 1e-9
t = t - t.iloc[0]

# -----------------------
# Position tracking plots
# -----------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes = ["x", "y", "z"]
for i, ax in enumerate(axes):
    axs[i].plot(t, df[f"h_des_{ax}"], label=f"h_des_{ax}")
    axs[i].plot(t, df[f"h_pred_{ax}"], label=f"h_pred_{ax}")
    axs[i].set_ylabel(f"{ax.upper()} position")
    axs[i].grid(True)
    axs[i].legend()

axs[-1].set_xlabel("Time [s]")
fig.suptitle("Desired vs Predicted Position")
plt.tight_layout()
plt.savefig(f"{out_dir}/position_tracking.png", dpi=300)
plt.close(fig)

# -----------------------
# Position error plots
# -----------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, ax in enumerate(axes):
    error = df[f"h_des_{ax}"] - df[f"h_pred_{ax}"]
    axs[i].plot(t, error, label=f"Error {ax}")
    axs[i].set_ylabel(f"{ax.upper()} error")
    axs[i].grid(True)
    axs[i].legend()

axs[-1].set_xlabel("Time [s]")
fig.suptitle("Position Tracking Error")
plt.tight_layout()
plt.savefig(f"{out_dir}/position_error.png", dpi=300)
plt.close(fig)

# -----------------------
# Torque vs Acceleration (time series)
# -----------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes = ["x", "y", "z"]
for i, ax in enumerate(axes):
    axs[i].plot(t, df[f"tau_{ax}"], label=f"tau_{ax}")
    axs[i].plot(t, df[f"acc_{ax}"], label=f"acc_{ax}")
    axs[i].set_ylabel(f"{ax.upper()} axis")
    axs[i].grid(True)
    axs[i].legend()

axs[-1].set_xlabel("Time [s]")
fig.suptitle("Torque vs Acceleration (Time Series)")
plt.tight_layout()
plt.savefig(f"{out_dir}/torque_vs_acc_time.png", dpi=300)
plt.close(fig)