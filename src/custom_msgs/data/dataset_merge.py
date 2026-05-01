import os
import numpy as np
import pandas as pd

def get_file_names(directory):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(directory)
        for f in files if f.endswith(".csv")
    ]

def read_csv_file(M, file_name):
    df = pd.read_csv(file_name)
    
    # Debug: Print column count and headers
    print(f"  Columns in file: {df.shape[1]}")
    print(f"  Headers: {list(df.columns)}")
    
    # Check if we have enough columns
    if df.shape[1] < 21:
        raise ValueError(f"File has only {df.shape[1]} columns, but needs at least 21. "
                        f"Missing columns for tau_x (18), tau_y (19), tau_z (20)")

    time  = df.iloc[:, 0].values        # string timestamp
    nsecs = df.iloc[:, 2].values        # int

    # Position
    x = df.iloc[:, 3]
    y = df.iloc[:, 4]
    z = df.iloc[:, 5]

    # Velocity
    x_dot = df.iloc[:, 6]
    y_dot = df.iloc[:, 7]
    z_dot = df.iloc[:, 8]

    # Acceleration
    x_ddot = df.iloc[:, 9]
    y_ddot = df.iloc[:,10]
    z_ddot = df.iloc[:,11]

    # Orientation
    roll  = df.iloc[:,12]
    pitch = df.iloc[:,13]
    yaw   = df.iloc[:,14]

    # Torques
    tau_x = df.iloc[:,18]
    tau_y = df.iloc[:,19]
    tau_z = df.iloc[:,20]

    # Previous torques
    tau_x_prev = np.roll(tau_x, 1)
    tau_y_prev = np.roll(tau_y, 1)
    tau_z_prev = np.roll(tau_z, 1)

    tau_x_prev[0] = 0
    tau_y_prev[0] = 0
    tau_z_prev[0] = 0

    # ---------- STATES (15) ----------
    states = np.column_stack([
        x, y, z,
        x_dot, y_dot, z_dot,
        x_ddot, y_ddot, z_ddot,
        roll, pitch, yaw,
        tau_x_prev, tau_y_prev, tau_z_prev
    ])

    # ---------- ACTIONS (3) ----------
    p_ddot = np.column_stack([x_ddot, y_ddot, z_ddot])
    tau    = np.column_stack([tau_x, tau_y, tau_z])

    actions = tau - M * p_ddot

    return time, nsecs, states, actions

def prepare_combined_csv():
    dataset_dir = "/home/control-lab/px4_ws/src/custom_msgs/dataset_drone_test"
    output_csv  = os.path.join(dataset_dir, "dataset_all_test.csv")
    nominal_mass = 1.0

    # Get all CSV files
    all_files = get_file_names(dataset_dir)
    
    # Filter for files starting with "data_"
    file_names = [
        f for f in all_files
        if os.path.basename(f).startswith("data_")
    ]

    if not file_names:
        print("❌ No CSV files found starting with 'data_'")
        return

    print(f"Found {len(file_names)} files to process\n")

    all_rows = []

    for file in file_names:
        print(f"Processing {file}")
        try:
            time, nsecs, states, actions = read_csv_file(nominal_mass, file)

            combined = np.hstack([
                time.reshape(-1, 1),      # string timestamps
                nsecs.reshape(-1, 1),
                states,
                actions
            ])

            all_rows.append(combined)
            print(f"  ✓ Successfully processed {combined.shape[0]} rows\n")
            
        except Exception as e:
            print(f"  ✗ Error processing file: {e}\n")
            continue

    if not all_rows:
        print("❌ No files were successfully processed")
        return

    # ---- combine all files ----
    final_data = np.vstack(all_rows)

    # ---- optional duplication logic ----
    N = final_data.shape[0]
    N_70 = int(0.7 * N)
    N_30 = N - N_70

    data_70 = final_data
    data_30 = final_data[:N_30]

    final_data = np.vstack([data_70, data_30])

    # ---- dataframe ----
    column_names = [
        "time",
        "nsecs",
        "x","y","z",
        "x_dot","y_dot","z_dot",
        "x_ddot","y_ddot","z_ddot",
        "roll","pitch","yaw",
        "tau_x_prev","tau_y_prev","tau_z_prev",
        "H_x","H_y","H_z"
    ]

    df = pd.DataFrame(final_data, columns=column_names)
    df.to_csv(output_csv, index=False)

    print("\n✅ Combined CSV created")
    print(f"📁 Saved at: {output_csv}")
    print(f"📏 Shape: {df.shape}")


if __name__ == "__main__":
    prepare_combined_csv()