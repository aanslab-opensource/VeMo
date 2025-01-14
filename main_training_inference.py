#----------------------------------------------------------------------------
# [Different code block]
# !pip install -r requirements.txt # uncomment this for colab-notebook usage
#----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pandas as pd

import VeMo
from   VeMo import VeMo_utils

tf.keras.backend.clear_session()
seed = 89
np.random.seed(seed=seed)
tf.random.set_seed(seed=seed)
init = tf.keras.initializers.GlorotNormal(seed=seed)
keras.utils.set_random_seed(seed=seed)
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

# PRE-PROCESSING
data_TRAIN = './dataset/train_set.csv'
data_VALID = './dataset/test_set.csv'
data_TEST  = './dataset/test_set.csv'

# data params
N_STEPS   = 100  # n-past samples to use in the model (last second)
N_FEATURE = 8    # number of net inputs


# filtering parameters for inputs and targets
in_cut = 45  # signals in inputs  are filtered at in_cut Hz
tg_cut = 45  # signals in targets are filtered at tg_cut Hz

# Gear is pratically always unfilterd
INPUT_FILTER = {
    "cutoff": [in_cut, in_cut, in_cut, 45, in_cut, in_cut, in_cut, in_cut],
    "sampling_freq": N_FEATURE * [100],
    "filter_order":  N_FEATURE * [8]
}

TARGET_FILTER = {
    "cutoff": [tg_cut, tg_cut, tg_cut, tg_cut],
    "sampling_freq": 4 * [100],
    "filter_order":  4 * [8]
}

columns_to_import = ['THROTTLE', 'BRAKE', 'STEERING', 'GEAR', 'AX', 'AY', 'YAWRATE', 'SPEED']

scaling_factors_inputs = {
    # driver controls
    'THROTTLE': 100,      # [-]
    'BRAKE':    100,      # [-]
    'STEERING': 250,      # [deg]
    'GEAR':       6,      # [-]
    'AX':     2 * 9.807,  # [m/s^2]
    'AY':     2 * 9.807,  # [m/s^2]
    'YAWRATE':   60,      # [deg/s]
    'SPEED':    280       # [km/h]
}

scaling_factors_targets = {
    # targets
    'AX':     2 * 9.807,  # [m/s^2]
    'AY':     2 * 9.807,  # [m/s^2]
    'YAWRATE':   60,      # [deg/s]
    'SPEED':    280       # [km/h]
}

dataset_TRAIN = pd.read_csv(data_TRAIN, delimiter=';', usecols=columns_to_import, header=0, skipinitialspace=True)
dataset_VALID = pd.read_csv(data_VALID, delimiter=';', usecols=columns_to_import, header=0, skipinitialspace=True)
dataset_TEST  = pd.read_csv(data_TEST,  delimiter=';', usecols=columns_to_import, header=0, skipinitialspace=True)

# Apply Butterworth filter to inputs
input_columns = ['THROTTLE', 'BRAKE', 'STEERING', 'GEAR', 'AX', 'AY', 'YAWRATE', 'SPEED']
dataset_TRAIN_inputs = VeMo_utils.butterworth(dataset_TRAIN[input_columns], INPUT_FILTER["cutoff"], INPUT_FILTER["sampling_freq"], INPUT_FILTER["filter_order"])
dataset_VALID_inputs = VeMo_utils.butterworth(dataset_VALID[input_columns], INPUT_FILTER["cutoff"], INPUT_FILTER["sampling_freq"], INPUT_FILTER["filter_order"])
dataset_TEST_inputs  = VeMo_utils.butterworth(dataset_TEST[ input_columns], INPUT_FILTER["cutoff"], INPUT_FILTER["sampling_freq"], INPUT_FILTER["filter_order"])

# Apply Butterworth filter to targets
target_columns = ['AX', 'AY', 'YAWRATE', 'SPEED']
dataset_TRAIN_targets = VeMo_utils.butterworth(dataset_TRAIN[target_columns], TARGET_FILTER["cutoff"], TARGET_FILTER["sampling_freq"], TARGET_FILTER["filter_order"])
dataset_VALID_targets = VeMo_utils.butterworth(dataset_VALID[target_columns], TARGET_FILTER["cutoff"], TARGET_FILTER["sampling_freq"], TARGET_FILTER["filter_order"])
dataset_TEST_targets  = VeMo_utils.butterworth(dataset_TEST[ target_columns], TARGET_FILTER["cutoff"], TARGET_FILTER["sampling_freq"], TARGET_FILTER["filter_order"])

# Scale and reshape data for the VeMo
x_TRAIN, _ = VeMo_utils.data_reshaper(VeMo_utils.scale_data(dataset_TRAIN_inputs,  scaling_factors_inputs),  N_STEPS)
_, y_TRAIN = VeMo_utils.data_reshaper(VeMo_utils.scale_data(dataset_TRAIN_targets, scaling_factors_targets), N_STEPS)

x_VALID, _ = VeMo_utils.data_reshaper(VeMo_utils.scale_data(dataset_VALID_inputs,  scaling_factors_inputs),  N_STEPS)
_, y_VALID = VeMo_utils.data_reshaper(VeMo_utils.scale_data(dataset_VALID_targets, scaling_factors_targets), N_STEPS)

x_TEST, _  = VeMo_utils.data_reshaper(VeMo_utils.scale_data(dataset_TEST_inputs,  scaling_factors_inputs),  N_STEPS)
_, y_TEST  = VeMo_utils.data_reshaper(VeMo_utils.scale_data(dataset_TEST_targets, scaling_factors_targets), N_STEPS)


print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

vemoNN = VeMo.VeMo(N_FEATURE, N_STEPS)

vemoNN.build_model()

vemoNN.train(x_TRAIN, y_TRAIN, x_VALID, y_VALID, epochs=3, batch_size=32)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

y_INFERENCE = vemoNN.inference(f'./VeMo_nn_{tg_cut}Hz.h5', x_TEST)
             #INPUT FILTER: V; MODEL TRAINED AT: V
np.savez(f'filtered_in{str(in_cut)}_model{str(tg_cut)}Hz_y_INFERENCE.npz', y_INFERENCE=y_INFERENCE)

np.savez(f'filtered_in{str(in_cut)}_model{str(tg_cut)}Hz_y_TEST.npz',      y_TEST=y_TEST)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

cut_start =  0
cut_end   = -1

errors = VeMo_utils.compute_errors(
    y_INFERENCE[cut_start:cut_end], 
    y_TEST[cut_start:cut_end], 
    f'filtered_in{str(in_cut)}_model{str(tg_cut)}Hz_computed_errors.npz', 
    scaling_factors_targets, 
    save=True
    )

abs_error_AX = np.max(np.abs((y_TEST[:, 0] * scaling_factors_targets['AX'])      - (y_INFERENCE[:, 0] * scaling_factors_targets['AX'])))
abs_error_AY = np.max(np.abs((y_TEST[:, 1] * scaling_factors_targets['AY'])      - (y_INFERENCE[:, 1] * scaling_factors_targets['AY'])))
abs_error_YR = np.max(np.abs((y_TEST[:, 2] * scaling_factors_targets['YAWRATE']) - (y_INFERENCE[:, 2] * scaling_factors_targets['YAWRATE'])))
abs_error_VX = np.max(np.abs((y_TEST[:, 3] * scaling_factors_targets['SPEED'])   - (y_INFERENCE[:, 3] * scaling_factors_targets['SPEED'])))
print(abs_error_AX)
print(abs_error_AY)
print(abs_error_YR)
print(abs_error_VX)

## EOF ##
