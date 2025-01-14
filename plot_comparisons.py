import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch
plt.rcParams.update({'font.size': 10}) #18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (8.3,11.7) # A4-vertical size
# plt.style.use('bmh')

#create the figure directory if not present
figure_dir_path = os.path.join(os.getcwd(), 'figures')
if not os.path.isdir(figure_dir_path): 
    os.makedirs(figure_dir_path)

y_labels = [r'$a_x$',r'$a_y$',r'$\dot\theta$',r'$v_x$']
y_units  = ['[m/s^2]','[m/s^2]','[deg/s]','[m/s]']
y_signals = ['AX','AY','YAWRATE','SPEED']
y_scaling = np.array([2*9.807, 2*9.807, 60, 280])

#set temporal indices to plot
idx_start =  0
idx_end   =  -1

# Parameters for PSD computation
fs = 100       # Sampling frequency in Hz
nperseg = 256  # Length of each segment for Welch's method
psd_units = ['$(m/s^2)^2/Hz$','$(m/s^2)^2/Hz$','$(deg/s)^2/Hz$','$(m/s)^2/Hz$']

cutfreqs = ['0.5', '5','25','45']
y_infr = {}
y_test = {}
for omega_cut in cutfreqs:    
    y_infr.update({omega_cut:np.load(f'./dataset/y_INFERENCE_{omega_cut}Hz.npz')['y_INFERENCE'][idx_start:idx_end,:] * y_scaling})
    y_test.update({omega_cut:np.load(f'./dataset/y_TEST_{     omega_cut}Hz.npz')['y_TEST'     ][idx_start:idx_end,:] * y_scaling})

time_vector = np.arange(0, len(y_test['0.5'])) / 100 # sampling_rate = 100 Hz


################################################################################
# FIG.9 (errors as function of time)

if True:

    fig, axs = plt.subplots(4, 1, figsize=(11.7, 8.3), sharex=True)

    for idx_omega, omega_cut in enumerate(cutfreqs):
        for idx_sig, (signal, ylabel, unit) in enumerate(zip(y_signals, y_labels, y_units)):
            abs_error = np.abs(y_infr[omega_cut][:, idx_sig] - y_test[omega_cut][:, idx_sig])
            rel_error = 100 * (abs_error) / np.max(np.abs(y_test[omega_cut][:, idx_sig]))
            axs[idx_sig].plot(time_vector, rel_error,    label=f'VeMo {omega_cut} Hz',   color=f'C{idx_omega}', linestyle='-', linewidth=0.5, alpha=0.5)
            axs[idx_sig].axhline(y=np.median(rel_error), label=f'median {omega_cut} Hz', color=f'C{idx_omega}', linestyle='--', linewidth=1.0)
            axs[idx_sig].axhline(y=np.mean(rel_error),   label=f'mean {omega_cut} Hz',   color=f'C{idx_omega}', linestyle='-.', linewidth=1.0)
            axs[idx_sig].set_ylabel('$\epsilon_{rel}$' + f'({ylabel})')
            axs[idx_sig].set_yscale('log')
            axs[idx_sig].grid(which='both')
            axs[idx_sig].set_ylim([0.1, 30])
            axs[idx_sig].set_yticks([0.1, 1, 10], labels=['0.1~\%', '1~\%', '10~\%'])

    for idx_sig in range(4): axs[idx_sig].grid(which='both')
    axs[-1].set_xlabel('Time $[s]$')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=4, fancybox=True, shadow=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig('./figures/relative_error_aggregated.pdf')
    plt.show()


################################################################################
# FIG.5 (histograms of the error)

if True:
        
    fig, axs = plt.subplots(1, 4, figsize=(8.3, 2.8), sharey=True)
    
    xlims = [25, 20, 40, 10]

    for omega_idx, omega_cut in enumerate(cutfreqs):
        
        errors = 100 * (y_test[omega_cut] - y_infr[omega_cut]) / np.max(np.abs(y_test[omega_cut]), axis=0)
        
        errors = np.abs(errors)

        for idx_sig, ax in enumerate(axs):
            
            error_std = np.std(errors[:, idx_sig])
            
            ax.hist(
                errors[:, idx_sig], 
                bins=np.linspace(0, xlims[idx_sig], 50), # np.linspace(0, np.max(np.abs(errors[:, idx_sig])), 30)
                color=f'C{omega_idx}', 
                histtype='step', 
                alpha=0.8, 
                log=True,
                density=False, 
                label=f'{omega_cut} Hz'
                )

            ax.set_xlabel(r'$\epsilon_{rel}($' + y_labels[idx_sig] + '$)$ \%')
                        
            if idx_sig == 0:
                ax.set_ylabel('Frequency')
            else:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False) # remove yaxis ticks for the second and third subplots

            ax.grid(True, linestyle='--', alpha=0.7)

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4, fancybox=True, shadow=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.165, top=0.85)
    # plt.savefig(f'./figures/histograms.pdf', format='pdf')
    plt.show()


################################################################################
# FIG.4 (Power Spectral Density - PSD)

if True:

    fig, axs = plt.subplots(3, 1, figsize=(8.3, 5), sharex=True)

    for idx_omega, omega_cut in enumerate(cutfreqs):

        for idx_sig, _ in enumerate(y_signals[:3]): # no SPEED

            f_test, psd_test = welch(y_test[omega_cut][:,idx_sig] * y_scaling[idx_sig],  fs, nperseg=nperseg)
            f_pred, psd_pred = welch(y_infr[omega_cut][:,idx_sig] * y_scaling[idx_sig],  fs, nperseg=nperseg)

            axs[idx_sig].semilogy(f_test, psd_test, label=f'Data {omega_cut} Hz', color=f'C{idx_omega}')
            axs[idx_sig].semilogy(f_pred, psd_pred, label=f'VeMo {omega_cut} Hz', color=f'C{idx_omega}', linestyle='--')

    for idx_sig in range(3): 
        axs[idx_sig].grid(True, linestyle='--', alpha=0.7)
        axs[idx_sig].set_ylabel('PSD$($' + y_labels[idx_sig] + '$)$\n\n' + psd_units[idx_sig])

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=4, fancybox=True, shadow=False)

    plt.xlabel('Frequency $[Hz]$')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09, top=0.88, hspace=0.04)
    # plt.savefig('./figures/psd_aggregated.pdf')
    plt.show()
