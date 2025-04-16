import json
import pandas as pd
import numpy as np
from enum import IntEnum, unique
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import butter, filtfilt




GRAVITY = 9.81


@unique
class DataVarIndex(IntEnum):
    '''A class that creates ids for the data.'''
    
    TIME = 0
    POS = 1
    VEL = 2
    ACC = 3
    JERK = 4
    SNAP = 5
    DES_POS = 6
    DES_VEL = 7
    DES_ACC = 8
    DES_JERK = 9
    DES_SNAP = 10

var_bounds = {
    DataVarIndex.POS: (-20, 20),
}


def load_data(filename):
    """Load the data from the csv file and return it as a numpy array."""
    # Read the data from the csv file, skipping the first row
    # and the last column has to be transformed using Status enum
    pd_data = pd.read_csv(filename)
    data = pd_data.to_numpy()

    # There may be a mismatch in the number of columns and the number of DataVarIndex. Add dummy values for the missing columns
    num_columns = len(DataVarIndex)
    num_data_columns = data.shape[1]
    dummy_data = np.zeros((data.shape[0], num_columns - num_data_columns))
    data = np.hstack((data, dummy_data))

    return data


class DataSmoother:
    """A class that smooths the data extracted from a .csv record file."""
    
    def __init__(self, file_path):
        # Create a dictionary to match the values with their derivatives
        # Zero-order level: position
        self.match_first_derivative = {
            DataVarIndex.POS: DataVarIndex.VEL,
            DataVarIndex.DES_POS: DataVarIndex.DES_VEL,
        }
        self.match_second_derivative = {    
            DataVarIndex.POS: DataVarIndex.ACC,
            DataVarIndex.DES_POS: DataVarIndex.DES_ACC,
        }
        self.match_third_derivative = {    
            DataVarIndex.POS: DataVarIndex.JERK,
            DataVarIndex.DES_POS: DataVarIndex.DES_JERK,
        }
        self.match_fourth_derivative = {    
            DataVarIndex.POS: DataVarIndex.SNAP,
            DataVarIndex.DES_POS: DataVarIndex.DES_SNAP,
        }

        self.match_cmd = {    
            DataVarIndex.POS: DataVarIndex.DES_POS,
            DataVarIndex.VEL: DataVarIndex.DES_VEL,
            DataVarIndex.ACC: DataVarIndex.DES_ACC,
            DataVarIndex.JERK: DataVarIndex.DES_JERK,
            DataVarIndex.SNAP: DataVarIndex.DES_SNAP,
        }

        self.derivatives = {
            1: self.match_first_derivative, 
            2: self.match_second_derivative,
            3: self.match_third_derivative,
            4: self.match_fourth_derivative,
        }

        # Load the data from the csv file
        self.data = load_data(file_path)
 
        self.flag_smoothed = False

        self.time = self.data[:, DataVarIndex.TIME] # array-like
        self.dt = np.mean(np.diff(self.time)) # numpy.diff -> calculate every time difference along the variable "time"
                                              # only useful within real-run, for simulation 'dt' is always a constant
        num_samples = self.time.shape[0] # function 'shape': dimensions of a array, shape[i] stand for the number of elements in dimension i
        self.time_interpolated = np.arange(num_samples) * self.dt + self.time[0] # theoritical time serious (not the real one), used for interpolation
        print("Resampling time interval: ", self.dt)
        print("Num samples: ", num_samples)

        self.splines = {}


    def smooth_data(self, indices_zero_order):
        """Smooth the data using cubic spline interpolation."""
        
        # Smooth the data
        for index in indices_zero_order: # indices stand for the name of data set, like pos_x, vel_x, cmd_thrust, etc.
            # Get the data that needs to be smoothed
            data = self.data[:, index]

            # use low-pass filter 
            N = 4
            Wn = 0.1
            b, a = butter(N, Wn, 'low')
            # apply butterfilter
            data_smoothed = filtfilt(b, a, data)
            self.data[:, index] = data_smoothed

            # Create a cubic spline interpolation function using 'splrep'
            # Parameters:
            #   k: highest degree for spline fitting, here k = 3 stand for cubic
            #   s: how smooth the curve is, default: s = 0.0
            # Return value: 
            #   tuple (t, c, k) -> tuple for representation of analytical description of interpolated curve
            data_spline = interpolate.splrep(self.time, data_smoothed, k=5, s=0.001) 

            # interpolated values on time points 'time_inetrpolated'
            data_interp = interpolate.BSpline(*data_spline)(self.time_interpolated)

            # store spline
            self.splines[index] = data_spline

            # Update the data
            self.data[:, index] = data_interp
        
        self.flag_smoothed = True # flip flag
    
        # Solve the derivatives using the splines
        self.solve_derivatives()

    def solve_derivatives(self):
        """replace the derivatives by taking the derivative of the spline interpolation."""

        if not self.flag_smoothed:
            raise ValueError("Data has to be smoothed first.")
        
        for der in self.derivatives.keys(): # der = 1 / 2 / 3 / 4
            match_derivative = self.derivatives[der]
            for index in match_derivative.keys():
                derivative_index = match_derivative[index]

                # Get the spline interpolation function
                spline = self.splines[index]
                # Take the derivative of the spline interpolation
                # Parameters:
                #   time points on which interpolated values are going to be solved
                #   '* data_spline' stand for tuple (t, c, k)        
                #   der: order of derivative, here are 1 or 2        
                # Return value: 
                #   set-order (1 / 2) derivative values of interpolated curve on time points 'time_inetrpolated'
                data_derivative_interp = interpolate.splev(self.time_interpolated, spline, der=der)

                # Optional: use low-pass filter for the derivatives
                #N = 4
                #Wn = 0.1
                #b, a = butter(N, Wn, 'low')
                # apply butterfilter
                #data_derivative_interp = filtfilt(b, a, data_derivative_interp)

                # Update the data
                self.data[:, derivative_index] = data_derivative_interp

    def save_data(self, file_path):
        """Save the smoothed data to a csv file."""

        if not self.flag_smoothed:
            raise ValueError("Data has to be smoothed first.")
        
        # Save the data to a csv file with DataVarIndex as the header
        pd_data = pd.DataFrame(self.data, columns=[var.name for var in DataVarIndex])
        pd_data.to_csv(file_path, index=False)
        
        print("Raw data has been smoothed and saved to: ", file_path)

        return file_path 
    
    def visualize(self):
        """Visualize the smoothed data."""

        if not self.flag_smoothed:
            raise ValueError("Data has to be smoothed first.")
        
        # Create a figure with 3 subplots individually for p/v/a, the x-axis is time and y-axis is the variables, 
        # in each subplot show curve before smooth in one color, after smooth in another color, cmd before smooth in dash line, cmd after smooth in dash line
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        for i, index in enumerate([DataVarIndex.POS, DataVarIndex.VEL, DataVarIndex.ACC]):
            # Plot the raw data
            axs[i].plot(self.time, self.data[:, index], label=DataVarIndex(index).name)
            # Plot the smoothed data
            axs[i].plot(self.time_interpolated, self.data[:, index], label=DataVarIndex(index).name + "_smoothed")
            # Plot the raw cmd data
            axs[i].plot(self.time, self.data[:, self.match_cmd(index)], label=DataVarIndex(self.match_cmd(index)).name)
            # Plot the smoothed cmd data
            axs[i].plot(self.time_interpolated, self.data[:, self.match_cmd(index)], label=DataVarIndex(self.match_cmd(index)).name + "_smoothed")
            axs[i].set_title(DataVarIndex(index).name)
            axs[i].legend()
        
        plt.show()









class ModelIdentifier:
    """A class that identifies the model parameters from the smoothed data."""
    
    def __init__(self, file_path_raw, smooth_indices_zero_order):

        self.smooth_indices_zero_order = smooth_indices_zero_order

        # Smooth the data
        self.smoother = DataSmoother(file_path_raw)
        self.smoother.smooth_data(self.smooth_indices_zero_order) # regulate time points, calculate data of smoothed curve and their 1 / 2 - derivatives

        # Save the smoothed data
        file_path_smoothed = file_path_raw.replace(".csv", "_smoothed.csv") # define name of new file
        self._smooth_data_path = self.smoother.save_data(file_path_smoothed)
        
        # Visualize the smoothed data
        self.smoother.visualize()

        self.params = {}
    
    @property
    def smooth_data_path(self): # get_method
        return self._smooth_data_path
    
    def identify_model(self, input_indices, output_indices):
        """Identify the model parameters from the smoothed data."""

        # initialize I/O of system dynamic, will be used in check_model
        self.input_indices = input_indices # DataVarIndex.DES_ACC
        self.output_indices = output_indices # DataVarIndex.JERK, DataVarIndex.ACC

        # Assume a model of the form:
        # acc_cmd = params[0] * jerk + params[1] * acc = [jerk, acc] * [params[0]; params[1]]
        # where acc_cmd = dot(vel_cmd), jerk = dot(acc), acc = dot(vel), vel = dot(pos)
        input_data = self.smoother.data[:, self.input_indices]
        #input_data = np.expand_dims(input_data, axis=1)
        output_data = None
        for output_index in self.output_indices:
            output_curr = self.smoother.data[:, output_index]
            output_curr = np.expand_dims(output_curr, axis=1)
            if output_data is None:
                output_data = output_curr
            else:
                output_data = np.hstack((output_data, output_curr))
        
        # Identify the linear model parameters using least squares
        self.params["k"] = np.linalg.lstsq(output_data, input_data, rcond=None)[0]

        # Save the identified model parameters
        identified_model_path = self.smooth_data_path.replace("_smoothed.csv", "_model.json")
        self.save_model(identified_model_path)


    def save_model(self, file_path):
        """Save the identified model parameters to a json file."""
        # Save the model parameters, data indices and the used data file names
        model_data = {
            "params_k": self.params["k"].tolist(),
            "input_indices": [var.name for var in self.input_indices],
            "output_indices": [var.name for var in self.output_indices],
            "smooth_indices": [var.name for var in self.smooth_indices],
            "data_file": self.smooth_data_path,
        }

        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=4)

        print("Model parameters saved to: ", file_path)
    
    
    def evaluate_model(self):
        """Evaluate the identified model by comparing the predicted and actual input data."""
        
        # use the identified model to predict the input data
        predicted_input = self.smoother.data[:, self.output_indices] @ self.params["k"]

        # retrieve the actual input data
        actual_input = self.smoother.data[:, self.input_indices]

        # Plot the predicted vs actual input data
        plt.figure(figsize=(8, 6))
        plt.scatter(actual_input, predicted_input, alpha=0.6, label="Predicted vs Actual", color='blue')
        plt.plot([actual_input.min(), actual_input.max()], [actual_input.min(), actual_input.max()], 
                linestyle='--', color='red', label="Ideal Fit (y=x)")
        
        # Calculate the error metrics
        residuals = actual_input - predicted_input
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        r2_score = 1 - (np.sum(residuals**2) / np.sum((actual_input - np.mean(actual_input))**2))

        # Add the error metrics to the plot
        plt.xlabel("Actual Input")
        plt.ylabel("Predicted Input")
        plt.title("Model Evaluation: Predicted vs Actual Input")
        plt.legend()
        plt.grid()

        # Print the error metrics
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"R² Score: {r2_score:.6f}")

        plt.show()