import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib as mpl
import matplotlib.cm as cm

#create the figure directory if not present
figure_dir_path = os.path.join(os.getcwd(), 'figures')
if not os.path.isdir(figure_dir_path): 
    os.makedirs(figure_dir_path)

def softmax(x):
    return 10*np.exp(x)/sum(np.exp(x))

# mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 10
mpl.rcParams['text.usetex'] = True

# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.sans-serif'] = 'Times'

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'

# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'DejaVu Serif'

mpl.style.use("_mpl-gallery")

########################################################################################################


columns_to_import = ['THROTTLE', 'BRAKE', 'STEERING', 'GEAR', 'AX', 'AY', 'YAWRATE', 'SPEED']

df_test  = pd.read_csv('./dataset/test_set.csv',  delimiter=';', usecols=columns_to_import)
df_train = pd.read_csv('./dataset/train_set.csv', delimiter=';', usecols=columns_to_import)


#######################################################################################################
# FIG.1 (train/test set plot) 

fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [0.8, 0.8, 1], 'wspace': 0.25})

# Set common color limits between colorbars
valmin, valmax = -50, 50

# Plot 1: AX vs SPEED
scatter1_train = axs[0].scatter(df_train['AX'], df_train['SPEED'], marker='o', s=1, c=df_train['YAWRATE'], cmap=cm.Blues, vmin=vmin, vmax=vmax)
scatter1_test = axs[0].scatter(df_test['AX'], df_test['SPEED'], marker='o', s=1, c=df_test['YAWRATE'], cmap=cm.Oranges, vmin=vmin, vmax=vmax)
axs[0].set_xlabel('$a_x$')
axs[0].set_ylabel('$v_x$')
axs[0].grid()

# Plot 2: AY vs SPEED
scatter2_train = axs[1].scatter(df_train['AY'], df_train['SPEED'], marker='o', s=1, c=df_train['YAWRATE'], cmap=cm.Blues, vmin=vmin, vmax=vmax)
scatter2_test = axs[1].scatter(df_test['AY'], df_test['SPEED'], marker='o', s=1, c=df_test['YAWRATE'], cmap=cm.Oranges, vmin=vmin, vmax=vmax)
axs[1].set_xlabel('$a_y$')
axs[1].grid()

# Remove y-axis ticks
axs[1].set_yticklabels([])
axs[1].tick_params(axis='y', which='both', left=False, right=False)

axs[1].set_position([axs[0].get_position().x1 + 0.02,
                     axs[1].get_position().y0,
                     axs[1].get_position().width,
                     axs[1].get_position().height])

# Plot 3: AY vs AX
scatter3_train = axs[2].scatter(df_train['AY'], df_train['AX'], marker='o', s=1, c=df_train['YAWRATE'], cmap=cm.Blues, vmin=vmin, vmax=vmax)
scatter3_test = axs[2].scatter(df_test['AY'], df_test['AX'], marker='o', s=1, c=df_test['YAWRATE'], cmap=cm.Oranges, vmin=vmin, vmax=vmax)
axs[2].set_xlabel('$a_y$')
axs[2].set_ylabel('$a_x$')
axs[2].grid()

cbar_ax1 = fig.add_axes([1 - 0.05, 0.15, 0.02, 0.7])
cbar1 = fig.colorbar(scatter3_train, cax=cbar_ax1, extend='both')
cbar1.set_label(r'$\dot{\theta}$ train', color='black', labelpad=-35)
cbar1.set_ticks(np.linspace(valmin, valmax, 5))
cbar1.set_ticks([])

cbar_ax2 = fig.add_axes([1, 0.15, 0.02, 0.7])
cbar2 = fig.colorbar(scatter3_test, cax=cbar_ax2, extend='both')
cbar2.set_label(r'$\dot{\theta}$ test', color='black', labelpad=-65)
cbar2.set_ticks(np.linspace(valmin, valmax, 5))

#plt.savefig('./figures/2d_dataset.png', format='png', dpi=300)
plt.show()


########################################################################################################
# FIG.8 (telemetry plot) 

# Load dataset
columns_to_import = ['THROTTLE', 'BRAKE', 'STEERING', 'GEAR', 'AX', 'AY', 'YAWRATE', 'SPEED']
csv_DATA = './dataset/test_set.csv'
dataset_TEST = pd.read_csv(csv_DATA, delimiter=';', usecols=columns_to_import)

# Configure plot style
mpl.rcParams.update({
    'font.size': 22,
    'figure.figsize': (8.3*2, 11.7*2)
})

time = np.linspace(0, len(dataset_TEST) / 100, len(dataset_TEST), endpoint=False)

state_labels = [r"$a_x$", r"$a_y$", r"$\dot{\theta}$", r"$v_x$"]
state_columns = ['AX', 'AY', 'YAWRATE', 'SPEED']

control_labels = [r"$u_t$", r"$u_b$", r"$u_s$", r"$u_g$"]
control_columns = ['THROTTLE', 'BRAKE', 'STEERING', 'GEAR']

height_ratios = [1, 1, 1, 1, 0.65, 0.65, 0.65, 0.65]

state_color = 'forestgreen'
control_color = 'red'

fig, axes = plt.subplots(8, 1, sharex=True, gridspec_kw={'height_ratios': height_ratios})

ylabel_x_position = -0.05

# Plot states
for ax, column, label in zip(axes[:4], state_columns, state_labels):
    ax.plot(time, dataset_TEST[column], label=label, color=state_color)
    ax.set_ylabel(label, labelpad=15)
    ax.yaxis.set_label_coords(ylabel_x_position, 0.5)
    ax.grid(True)

# Plot control actions
for i, (ax, column, label) in enumerate(zip(axes[4:], control_columns, control_labels)):
    ax.plot(time, dataset_TEST[column], label=label, color=control_color)
    ax.set_ylabel(label, labelpad=15)
    ax.yaxis.set_label_coords(ylabel_x_position, 0.5)
    ax.grid(True)

    if column == 'GEAR':
        gear_ticks = range(int(dataset_TEST[column].min()), int(dataset_TEST[column].max()) + 1)
        ax.set_yticks(gear_ticks)

# Add sector vertical lines to all control axes
for ax in axes[4:]:
    for start, end, _ in sectors:
        ax.axvline(x=start, color='black', linestyle='--', linewidth=1.3)
        ax.axvline(x=end, color='black', linestyle='--', linewidth=1.3)

OS = 6.9300  # Offset for resetting time, respect to MoTec Logs
sectors = [
    (0, 105.34 - OS, "Warm Up"),
    (105.34 - OS, 211.08 - OS, "Hot Lap"),
    (211.08 - OS, 322.35 - OS, "Sine Waves"),
]

for start, end, label in sectors:
    axes[0].axvline(x=start, color='black', linestyle='--', linewidth=1.3) 
    axes[0].axvline(x=end, color='black', linestyle='--', linewidth=1.3)    
    axes[0].text((start + end) / 2,  
                 axes[0].get_ylim()[1] * 1.1,  
                 label, color='black', fontsize=18, ha='center', va='bottom', weight='bold')

for ax in axes[1:]:
    for start, end, _ in sectors:
        ax.axvline(x=start, color='black', linestyle='--', linewidth=1.3)
        ax.axvline(x=end, color='black', linestyle='--', linewidth=1.3)

axes[-1].set_xlabel(r"Time [$s$]")
plt.tight_layout(rect=[0, 0, 1, 0.95])  
#plt.savefig('./figures/telemetry.pdf', format='pdf')
plt.show()


