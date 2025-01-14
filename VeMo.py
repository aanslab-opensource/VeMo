########################################################################################
# IMPORTS

import os  
import numpy        as np
import pandas       as pd
from   scipy.signal import butter, filtfilt
from   matplotlib   import pyplot as plt
from   matplotlib   import colors as mcolors
import tensorflow   as tf

import keras.backend        as K
from   keras.models         import Model
from   keras.layers         import Input, GRU, Dense, Dropout
from   keras.initializers   import GlorotUniform
from   keras.models         import load_model

#########################################################################################
# CUSTOM OBJECTS FOR VeMo

def ax_loss(y_true, y_pred):
    mae_loss_ax = K.mean(K.abs(y_true[:, 0]    - tf.squeeze(y_pred) + 1e-10), axis=-1) # Mean Absolute Error
    mse_loss_ax = K.mean(K.square(y_true[:, 0] - tf.squeeze(y_pred) + 1e-10), axis=-1) # Mean Square Error
    loss_ax = (1.0 * mae_loss_ax + 0 * mse_loss_ax )                                   # Mixing for regularization in training
    return loss_ax

def ay_loss(y_true, y_pred):
    mae_loss_ay = K.mean(K.abs(y_true[:, 1]    - tf.squeeze(y_pred) + 1e-10), axis=-1)
    mse_loss_ay = K.mean(K.square(y_true[:, 1] - tf.squeeze(y_pred) + 1e-10), axis=-1)
    loss_ay = (1 * mae_loss_ay + 0. * mse_loss_ay) 
    return loss_ay

def yr_loss(y_true, y_pred):
    mae_loss_yr = K.mean(K.abs(y_true[:, 2]    - tf.squeeze(y_pred) + 1e-10), axis=-1)
    mse_loss_yr = K.mean(K.square(y_true[:, 2] - tf.squeeze(y_pred) + 1e-10), axis=-1)
    loss_yr = (1 * mae_loss_yr + 0 * mse_loss_yr) 
    return loss_yr

def vx_loss(y_true, y_pred):
    mae_loss_vx = K.mean(K.abs(y_true[:, 3]    - tf.squeeze(y_pred) + 1e-10), axis=-1)
    mse_loss_vx = K.mean(K.square(y_true[:, 3] - tf.squeeze(y_pred) + 1e-10), axis=-1)
    loss_vx = (1 * mae_loss_vx + 0 * mse_loss_vx) 
    return loss_vx

#########################################################################################
# VeMo CORE CLASS

class VeMo:

    def __init__(self, n_features=9, n_steps=50):
        self.n_features    = n_features
        self.n_steps       = n_steps
        self.VeMo_Net      = None


    def build_model(self, seed=89):
        
        input_layer = Input(shape=(self.n_steps, self.n_features), name='input_layer')
   
        #------------- MODEL BODY -----------------------------------------------
        encode_1 = GRU(30, activation='elu', kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=True, name='encode_GRU1')(input_layer)
        encode_2 = GRU(20, activation='elu', kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=True, name='encode_GRU2')(encode_1)
        encode_3 = GRU(10,  activation='elu', kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=True, name='encode_GRU3')(encode_2)

        branch_ax_0 = Dropout(0.05, seed=seed, name='Drop1_ax')(encode_3)
        branch_ax_1 = GRU(  5, activation='tanh',           kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=True,  name='GRU1_ax')(branch_ax_0)
        branch_ax_2 = GRU(  5, activation='tanh',           kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=False, name='GRU2_ax')(branch_ax_1)
        branch_ax_3 = Dense(5, activation='LeakyReLU',      kernel_initializer=GlorotUniform(seed=seed), use_bias=False,                         name='Dense1_ax')(branch_ax_2)

        branch_ay_0 = Dropout(0.05, seed=seed, name='Drop1_ay')(encode_3)
        branch_ay_1 = GRU(  5, activation='tanh',           kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=True,  name='GRU1_ay')(branch_ay_0)
        branch_ay_2 = GRU(  5, activation='tanh',           kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=False, name='GRU2_ay')(branch_ay_1)
        branch_ay_3 = Dense(5, activation='LeakyReLU',      kernel_initializer=GlorotUniform(seed=seed), use_bias=False,                         name='Dense1_ay')(branch_ay_2)

        branch_yr_0 = Dropout(0.05, seed=seed, name='Drop1_yr')(encode_3)
        branch_yr_1 = GRU(  5, activation='tanh',           kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=True,  name='GRU1_yawrate')(branch_yr_0)
        branch_yr_2 = GRU(  5, activation='tanh',           kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=False, name='GRU2_yawrate')(branch_yr_1)
        branch_yr_3 = Dense(5, activation='LeakyReLU',      kernel_initializer=GlorotUniform(seed=seed), use_bias=False,                         name='Dense1_yawrate')(branch_yr_2)
        
        branch_vx_0 = Dropout(0.05, seed=seed, name='Drop1_vx')(encode_3)
        branch_vx_1 = GRU(  5, activation='tanh',           kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=True,  name='GRU1_vx')(branch_vx_0)
        branch_vx_2 = GRU(  5, activation='tanh',           kernel_initializer=GlorotUniform(seed=seed), use_bias=False, return_sequences=False, name='GRU2_vx')(branch_vx_1)
        branch_vx_3 = Dense(5, activation='LeakyReLU',      kernel_initializer=GlorotUniform(seed=seed), use_bias=False,                         name='Dense1_vx')(branch_vx_2)
        #------------- MODEL OUTPUTS -----------------------------------------------
        output_ax = Dense(1, activation='linear', use_bias=False, name='output_ax'     )(branch_ax_3)
        output_ay = Dense(1, activation='linear', use_bias=False, name='output_ay'     )(branch_ay_3)
        output_yr = Dense(1, activation='linear', use_bias=False, name='output_yr'     )(branch_yr_3)
        output_vx = Dense(1, activation='linear', use_bias=False, name='output_vx'     )(branch_vx_3)
        
        #------------- MODEL COMPILING OPTIONS -----------------------------------------------
        self.VeMo_Net = Model(inputs=[input_layer], outputs=[output_ax, output_ay, output_yr, output_vx])

        self.VeMo_Net.compile(
            optimizer = 'adam',
            loss = {
                'output_ax':      ax_loss, 
                'output_ay':      ay_loss, 
                'output_yr':      yr_loss,
                'output_vx':      vx_loss
                }
            )

        print(self.VeMo_Net.summary())
        
        return self.VeMo_Net


    # @staticmethod
    # def plot_nn(VeMo_Net):
        # plot_model(VeMo_Net, to_file='VeMo_Net.png', show_shapes=True, show_layer_names=True, show_layer_activations=True, rankdir='LR', dpi=300)
        # plt.figure(dpi=300)
        # img = plt.imread('VeMo_Net.png')
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()        


    def train(self, input_data, target_data, validation_data, validation_target, epochs=8, batch_size=64, save_hist=True, file_name='VeMo_Net.h5'):
        input_data_main     = input_data
        val_input_data_main = validation_data
        val_target_data     = validation_target

        history = self.VeMo_Net.fit(
            [input_data_main], 
            target_data,
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(
                [val_input_data_main], 
                val_target_data
                )
            )

        self.VeMo_Net.save(f"./{file_name}")
        
        if save_hist is True:
            np.savez('training_history.npz', **history.history)
            print('Training History saved as \'training_history.npz')
        
        return history.history


    @staticmethod
    def inference(path_to_VeMo_Net, x_test, save=False):
        
        custom_settings = {
            'ax_loss':      ax_loss,
            'ay_loss':      ay_loss,
            'yr_loss':      yr_loss, 
            'vx_loss':      vx_loss # losses
            }

        VeMo_Net = load_model(
            path_to_VeMo_Net, 
            custom_objects=custom_settings, 
            safe_mode=False
            )
        
        y_pred_loaded = np.array(VeMo_Net.predict(x_test))

        y_pred_loaded = y_pred_loaded[:,:,0].T

        if save is True:
            print('\nfrom VeMo: NN Simulation Complete!, signals order: [ax, ay, yawrate, vx]\n')
            np.savez("NN_Simulation_Out.npz", y_pred_loaded = y_pred_loaded)
            print('Simulation Output saved as \'nn_simulation_out.npz\'')

        return y_pred_loaded
    
#########################################################################################
# UTILS FUNCTIONS

class VeMo_utils:
    
    @staticmethod
    def data_reshaper(series, n_steps):

        x, y = [], []
        
        for i in range(len(series) - n_steps):
            
            end_idx = i + n_steps  # Index for the end of the current sequence
            
            # Inputs: past driver controls and dynamics
            seq_x = np.array(series[i:end_idx])  # All columns for n_steps
            
            # Target: next dynamics (AX, AY, YAW-RATE, SPEED, at end_ix)
            seq_y = np.array(series[end_idx, -4:])  # Last 4 columns at the next step
            
            x.append(seq_x)
            y.append(seq_y)
        
        x = np.array(x).reshape(-1, n_steps, series.shape[1])  # Include all input features
        y = np.array(y)  # Targets remain 2D
        
        return x, y

    
    @staticmethod
    def butterworth(data, cut_off_list, fs_list, order_list):

        filtered_signals = {}

        min_length = float('inf')  # Track the minimum length after filtering

        for i, col in enumerate(data.columns):

            cut_off = cut_off_list[i]

            fs = fs_list[i]

            order = order_list[i]

            nyq = 0.5 * fs  # Nyquist frequency

            normal_cutoff = cut_off / nyq

            b, a = butter(order, normal_cutoff, btype='low', analog=False)

            filtered_col = filtfilt(b, a, data[col])

            filtered_signals[col] = filtered_col

            min_length = min(min_length, len(filtered_col))

        # Trim all signals to the shortest length to ensure coherent dimensions
        trimmed_signals = {col: filtered_signals[col][:min_length] for col in data.columns}

        return pd.DataFrame(trimmed_signals)


    @staticmethod
    def scale_data(df, scaling_factors):  

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        missing_columns = [col for col in scaling_factors if col not in df.columns]
        
        if missing_columns:
            raise KeyError(f"Columns missing in DataFrame: {', '.join(missing_columns)}")
    
        for column, factor in scaling_factors.items():
            df[column] = df[column] / factor

        data_array = df.to_numpy()

        nan_indices = np.isnan(data_array)
        
        if np.any(nan_indices):
            print("\nfrom VeMo: There are NaN values in the dataset.")
            print("from VeMo: Indices of NaN values:")
            print(np.argwhere(nan_indices))

        return data_array


    @staticmethod
    def compute_errors(y_pred, y_true, file_name, scaling_factors=None, save=False): 

        def rmse(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())

        if scaling_factors is None:
            scaling_factors = {'AX': 1.0, 'AY': 1.0, 'YAWRATE': 1.0, 'SPEED' : 1.0}

        errors_npz = {}

        for i, key in enumerate(['AX', 'AY', 'YAWRATE', 'SPEED']):

            abs_error = np.abs((y_true[:, i] * scaling_factors[key]) - (y_pred[:, i] * scaling_factors[key]))
            
            rel_error = 100 * (abs_error) / np.max(np.abs((y_true[:, i] * scaling_factors[key])))

            errors_npz[key] = {
                'Median Relative Error': np.median(rel_error), 
                'Mean Relative Error':   np.mean(rel_error),
                'Median Absolute Error': np.median(abs_error),
                'Mean Absolute Error':   np.mean(abs_error),
                'RMSE': rmse((y_pred[:, i]* scaling_factors[key]), (y_true[:, i] * scaling_factors[key])) 
            }

        if save is True:
            print('\nfrom VeMo: NN Errors Computed!, signals order: [Median Relative Error, Mean Relative Error, Median Absolute Error, Mean Absolute Error, RMSE]\n')
            np.savez(file_name, **errors_npz) 
            print('Results saved as \'computed_errors.npz')

        return errors_npz


    @staticmethod
    def rescale_output(y_pred_loaded, scaling_factors):
        y_pred_loaded[:, 0] = y_pred_loaded[:, 0] * scaling_factors['AX']
        y_pred_loaded[:, 1] = y_pred_loaded[:, 1] * scaling_factors['AY']
        y_pred_loaded[:, 2] = y_pred_loaded[:, 2] * scaling_factors['YAWRATE']
        y_pred_loaded[:, 3] = y_pred_loaded[:, 3] * scaling_factors['SPEED']
        return y_pred_loaded











































########################################################################################
# VEHICLE PRE-MODEL BUILDER 

# class VeMo_premodel:

#     def __init__(self, g=9.80665, v_max=70):
#         self.g = g          # Standard gravity
#         self.v_max = v_max  # Maximum velocity [m/s]

#     def ax_estimator(self, gear, throttle_percentage, brake_percentage, A=0.0109, B=1.6, C=0.02):  

#         def f_aux_ax(gear, throttle_percentage, brake_percentage, A, B, C):
#             if brake_percentage > 0:
#                 return -(C * self.g) * brake_percentage
#             else:
#                 return ((A * self.g) * throttle_percentage) - (self.g * np.log10(B * gear))

#         def f_ax(gear, throttle_percentage, brake_percentage, A, B, C):
#             if gear < 1:
#                 return 0
#             else:
#                 return f_aux_ax(gear, throttle_percentage, brake_percentage, A, B, C)
        
#         ax_model = f_ax(gear, throttle_percentage, brake_percentage, A, B, C)
#         return ax_model 
    
#     def ay_estimator(self, gear, steering_angle, D=2.05, E=0.5, F=0.015):
#         ay_model = D * np.tanh(gear) * np.tanh(E * gear * np.sinh(F * steering_angle)) * self.g
#         return ay_model
    
#     def yawrate_estimator(self, gear, steering_angle, G=0.005):
#         yawrate_model = (self.v_max * gear) * np.tanh(G * steering_angle)
#         return yawrate_model
    
#     def compute_premodel(self, input_data):

#         if not isinstance(input_data, pd.DataFrame):
#             raise ValueError("Input data must be a pandas DataFrame")
        
#         throttle = input_data['THROTTLE'].to_numpy()
#         brake    = input_data['BRAKE'].to_numpy()
#         steering = input_data['STEERING'].to_numpy()
#         gear     = input_data['GEAR'].to_numpy()

#         ax_computed = np.zeros(len(input_data))
#         ay_computed = np.zeros(len(input_data))
#         yawrate_computed = np.zeros(len(input_data))

#         for i in range(len(input_data)):
#             ax_computed[i] = self.ax_estimator(gear[i], throttle[i], brake[i])
#             ay_computed[i] = self.ay_estimator(gear[i], steering[i])
#             yawrate_computed[i] = self.yawrate_estimator(gear[i], steering[i])

#         result_df = input_data.copy()
#         result_df['AX_m'] = ax_computed
#         result_df['AY_m'] = ay_computed
#         result_df['YAWRATE_m'] = yawrate_computed

#         return result_df

#     def process_and_add_models(self, input_data):
#         # Compute the premodel values and add them to the dataset
#         result_df = self.compute_premodel(input_data)
        
#         # Reorder columns to insert computed results after the first four columns
#         columns = input_data.columns.tolist()
#         driver_controls = columns[:4]
#         targets         = columns[4:]
        
#         # New order with computed columns inserted after the first four columns
#         new_order = driver_controls + ['AX_m', 'AY_m', 'YAWRATE_m'] + targets
        
#         # Reorder the DataFrame
#         result_df = result_df[new_order]
        
#         return result_df

#########################################################################################
# EOF
