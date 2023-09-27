import numpy as np
import pandas as pd

def extract_amp_phase(path, names = ('amp_no_name.csv', 'phase_no_name.csv')): 
    '''Returns and save 2 different datasets, 
    1 for the amplitude 1 for the phase, both with 52 features.
    It combines the I/Q samples to produce the results.
    '''
    data_0 = pd.read_csv(path)
    data_0 = data_0['CSI_DATA'].str.strip('[]').apply(lambda x: [int(num) for num in x.split()])

    def compute_amplitude(arr):
        arr = np.array(arr)
        return np.sqrt(np.sum(arr.reshape(-1, 2)**2, axis=1))

    def compute_atan2(arr):
        arr = np.array(arr)
        imaginary, real = arr[::2], arr[1::2] # imaginary = y = Q and real = x = I of I/Q-Sampling
        return np.rad2deg(np.arctan2(imaginary, real)) # argument order is (y,x) NOT (x,y)

    amp_data = data_0.apply(compute_amplitude)
    phase_data = data_0.apply(compute_atan2)
    amp_data = pd.DataFrame(amp_data.tolist(), columns=[f'subc{i+1}' for i in range(64)]).iloc[:,6:59].drop('subc33',axis=1)  # remove guard subcarriers
    phase_data = pd.DataFrame(phase_data.tolist(), columns=[f'subc{i+1}' for i in range(64)]).iloc[:,6:59].drop('subc33', axis=1)  # remove guard subcarriers
    amp_data.to_csv('..\Data\\' + names[0], index=False)
    phase_data.to_csv('..\Data\\' + names[1], index=False)
    return amp_data, phase_data

def make_alternating(amp,pha):
    ''' Combine amplitude and phase in an alternating way like: 
    [a1,p1, a2,p2, ..., a64,p64] where aN and pN are
    amplitude and phase of subcarrier N
    '''
    ampNP = amp.to_numpy()[:, :, np.newaxis]
    phaNP = pha.to_numpy()[:, :, np.newaxis]
    data = np.concatenate((ampNP, phaNP), axis=2)
    data = data.reshape(data.shape[0],-1)
    return data

def phase_sanitization_inRange(row):
    ''' Use: YourDataFrame.apply(phase_sanitization, axis=1)'''
    a = (row[-1] - row[0]) / 52
    b = row.mean()
    half_1 = np.arange(7, 33)
    half_2 = np.arange(34, 60)
    mi_values = np.concatenate((half_1, half_2))
    row = row - a * (mi_values - 33) - b #stop here if you want no bounds
    # Use modulo to wrap angles into the [0, 2π) range
    bounded_angles = row % (2 * np.pi)
    # Ensure the result is positive
    bounded_angles[bounded_angles < 0] += 2 * np.pi
    return bounded_angles

def convert_to_bounded_angle(angles):
    ''' Use: YourDataFrame.apply(convert_to_bounded_angle, axis=1)'''
    # Use modulo to wrap angles into the [0, 2π) range
    bounded_angles = angles % (2 * np.pi)
    # Ensure the result is positive
    bounded_angles[bounded_angles < 0] += 2 * np.pi
    return bounded_angles


def MAD_filter(df, window_size=500, stride=None, thresh=3):
    if stride == None:
        stride = window_size
    # Create an empty DataFrame to store the cleaned data
    cleaned_df = df.copy()
    # Loop through each column (feature) in the DataFrame
    for column in df.columns:
        for i in range(0, len(df) - window_size + 1, stride):
            # Select the current window
            window = df[column].iloc[i:i+window_size]
            # Calculate the median and MAD for the window
            median = window.median()
            mad = np.median(np.abs(window - median))
            # Define a threshold for outlier detection (e.g., 3 times MAD)
            threshold = thresh * mad
            # Identify outliers within the window
            outliers = np.abs(window - median) > threshold
            # Replace outliers with the mean of the non-outlier values in the window
            for j in range(len(window)):
                if outliers.iloc[j]:
                    # Find the indices of previous and next non-outlier values within the window
                    prev_non_outlier = j - 1
                    next_non_outlier = j + 1
                    while prev_non_outlier >= 0 and outliers.iloc[prev_non_outlier]:
                        prev_non_outlier -= 1
                    while next_non_outlier < len(window) and outliers.iloc[next_non_outlier]:
                        next_non_outlier += 1
                    # Replace the outlier with the mean of the non-outlier values in the window
                    if prev_non_outlier >= 0 and next_non_outlier < len(window):
                        cleaned_df.at[i + j, column] = (
                            window.iloc[prev_non_outlier] + window.iloc[next_non_outlier]
                        ) / 2
    return cleaned_df

def create_tensor(df, winsize = 200, overlap =100):
    num_images = (len(df) - overlap) // (winsize - overlap)  # Calculate the number of images
    # Initialize aan empty NumPy array to store the image data
    image_data = np.empty((num_images, winsize, df.shape[1]))
    # Create overlapping slices and populate the image data
    for i in range(num_images):
        start = i * (winsize - overlap)
        end = start + winsize
        image_slice = df.iloc[start:end, :]
         # Store the slice data in the image_data 3D array
        image_data[i, :, :] = image_slice.values
    # image_data now contains your images as a 3D NumPy array (num_images, window_size, num_columns)
    return image_data



def replace_outliers_std(df, std_threshold=3, inplace=False):
    if not inplace:
        df = df.copy()
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        # Identify outliers
        outliers = (df[col] - mean).abs() > std_threshold * std
        # Iterate through the DataFrame and replace outliers
        for i in range(1, len(df) - 1):
            if outliers.iloc[i]:
                # Find the previous and next non-outlier indices
                prev_index = i - 1
                next_index = i + 1
                while prev_index >= 0 and outliers.iloc[prev_index]:
                    prev_index -= 1
                while next_index < len(df) and outliers.iloc[next_index]:
                    next_index += 1
                # Calculate the average of previous and next non-outliers
                avg_value = (df.iloc[prev_index][col] + df.iloc[next_index][col]) / 2
                # Replace the outlier with the average
                df.at[i, col] = avg_value
    return df

def replace_outliers_iqr(df):
    # Create an empty DataFrame to store the cleaned data
    cleaned_df = df.copy()
    outNum = []
    # Loop through each column (feature) in the DataFrame
    for column in df.columns:
        # Calculate Q1 and Q3 for the current feature
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        # Calculate the IQR for the current feature
        IQR = Q3 - Q1
        # Define lower and upper bounds for outlier detection
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Identify outliers for the current feature
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        # Replace outliers with the mean of the previous and next non-outlier values
        for i in range(len(df)):
            if outliers.iloc[i]:
                # Find the indices of previous and next non-outlier values
                prev_non_outlier = i - 1
                next_non_outlier = i + 1
                # Find the previous non-outlier value
                while prev_non_outlier >= 0 and outliers.iloc[prev_non_outlier]:
                    prev_non_outlier -= 1
                # Find the next non-outlier value
                while next_non_outlier < len(df) and outliers.iloc[next_non_outlier]:
                    next_non_outlier += 1
                # Replace the outlier with the mean of the previous and next non-outlier values
                if prev_non_outlier >= 0 and next_non_outlier < len(df):
                    cleaned_df.at[i, column] = (df.at[prev_non_outlier, column] + df.at[next_non_outlier, column]) / 2
    return cleaned_df

import numpy as np

def hampel_filtering(df, window_size, thresh=3, smoothing_factor=0.9):
    # Create an empty DataFrame to store the cleaned data
    cleaned_df = df.copy()
    
    # Calculate half window size (since the window is centered, it's half to the left and half to the right)
    half_window = (window_size - 1) // 2
    
    # Loop through each column (feature) in the DataFrame
    for column in cleaned_df.columns:
        # Skip points that are within the initial half window size and final half window size
        for i in range(half_window, len(cleaned_df) - half_window): # [1,2,3,4,5,6]
            # Determine the window indices considering edges
            start_index = i - half_window
            end_index = i + half_window + 1
            
            # Select the current window
            window = cleaned_df[column].iloc[start_index:end_index]
            
            # Calculate the median and MAD for the window
            median = window.median()
            mad = np.median(np.abs(window - median))
            
            # Define a threshold for outlier detection (e.g., 3 times MAD)
            threshold = thresh * mad
            
            # Identify if the current point is an outlier based on the MAD method
            is_outlier = np.abs(cleaned_df[column].iloc[i] - median) > threshold
            
            # If it's an outlier, replace it with an estimate based on exponential smoothing
            if is_outlier:
                # Initialize smoothed value with the first point in the window
                smoothed_value = cleaned_df[column].iloc[start_index]
                for j in range(start_index, i):
                    # Compute exponential smoothing
                    smoothed_value = smoothing_factor * cleaned_df[column].iloc[j] + (1 - smoothing_factor) * smoothed_value
                cleaned_df.at[i, column] = smoothed_value
    
    return cleaned_df