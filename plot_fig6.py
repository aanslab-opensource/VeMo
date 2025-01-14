# FIG.6 (yaw rate comparison with filtered/noisy inputs)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.sans-serif'] = 'DejaVu Serif'

#create the figure directory if not present
figure_dir_path = os.path.join(os.getcwd(), 'figures')
if not os.path.isdir(figure_dir_path): 
    os.makedirs(figure_dir_path)

# Scaling factor for yaw rate
scaling_factors_targets = {
    'YAWRATE': 60  # [deg/s]
}

# Load data
idx_start = 0
idx_end = 5000
sampling_rate = 100  # Hz
time = np.arange(idx_start, idx_end) / sampling_rate

# Load yaw rate data from the files
y_INFERENCE_noise   = np.load('./dataset/y_INFERENCE_0.5Hz_inputs_45Hz.npz')['y_INFERENCE'][cut_start:cut_end, :]
y_INFERENCE_nonoise = np.load('.dataset//y_INFERENCE_0.5Hz.npz')['y_INFERENCE'][cut_start:cut_end, :]
y_TEST = np.load('.dataset//y_TEST_0.5Hz.npz')['y_TEST'][cut_start:cut_end, :]

# Extract yaw rate
yaw_rate_inference_nonoise = y_INFERENCE_nonoise[:, 2] * scaling_factors_targets['YAWRATE']
yaw_rate_inference_noise   = y_INFERENCE_noise[:, 2]   * scaling_factors_targets['YAWRATE']
yaw_rate_test = y_TEST[:, 2] * scaling_factors_targets['YAWRATE']

# Plot the data
plt.figure(figsize=(8.3, 5.5))
plt.plot(time, yaw_rate_test,              label='Data 0.5 Hz',          linestyle='-',  color='black',     linewidth=2)
plt.plot(time, yaw_rate_inference_noise,   label='VeMo 0.5 Hz @ 45 Hz',  linestyle='--', color='royalblue', linewidth=2)
plt.plot(time, yaw_rate_inference_nonoise, label='VeMo 0.5 Hz @ 0.5 Hz', linestyle='--', color='crimson',   linewidth=2)

plt.xlabel(r"Time [$s$]")
plt.ylabel(r"$\dot\theta$")
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# plt.savefig("./figures/yaw_rate_noise.pdf", format="pdf")
plt.show()
