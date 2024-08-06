import numpy as np
import pandas as pd

def extract_amp_phase(path, names = ('amp_no_name.csv', 'phase_no_name.csv')): 
    '''Returns and save 2 different datasets, 1 for the amplitude 1 for the phase, both with 52 features.
       It combines the I/Q samples to produce the results, ignoring empty subcarriers.
       Parameters:
       path: the path of .csv file with csi data
       names: how to save the amp and phase files, the path will be starting from "..\Data\DataClean\\"
    '''
    data_0 = pd.read_csv(path)
    data_0 = data_0['CSI_DATA'].str.strip('[]').apply(lambda x: [int(num) for num in x.split()])

    def compute_amplitude(arr):
        arr = np.array(arr)
        return np.sqrt(np.sum(arr.reshape(-1, 2)**2, axis=1))

    def compute_atan2(arr):
        arr = np.array(arr)
        imaginary, real = arr[::2], arr[1::2] # imaginary = y = Q and real = x = I of I/Q-Sampling
        return np.arctan2(imaginary, real) # argument order is (y,x) NOT (x,y), returns Radians

    amp_data = data_0.apply(compute_amplitude)
    phase_data = data_0.apply(compute_atan2)
    amp_data = pd.DataFrame(amp_data.tolist(), columns=[f'subc{i-32}' for i in range(64)]).iloc[:,6:59].drop('subc0',axis=1)  # remove guard subcarriers
    phase_data = pd.DataFrame(phase_data.tolist(), columns=[f'subc{i-32}' for i in range(64)]).iloc[:,6:59].drop('subc0', axis=1)  # remove guard subcarriers
    amp_data.to_csv('..\Data\DataClean\\' + names[0], index=False)
    phase_data.to_csv('..\Data\DataClean\\' + names[1], index=False)
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
    ''' Use: YourDataFrame.apply(phase_sanitization_inRange, axis=1)
        It applies the calibration procedure to the phase dataframe
        and map the angles into [0, 2pi) range
    '''
    a = (row[-1] - row[0]) / 52
    b = row.mean()
    mi_values = np.concatenate((np.arange(-26, 0),np.arange(1,27)))
    row = row - (a * mi_values) - b # stop here if you want no bounds
    # Use modulo to wrap angles into the [0, 2π) range
    bounded_angles = row % (2 * np.pi)
    # Ensure the result is positive
    bounded_angles[bounded_angles < 0] += 2 * np.pi
    return bounded_angles

def hampel_filtering(df, window_size, thresh=3, smoothing_factor=0.9):
    ''' Performs hampel filtering with a sliding window and substitutes the outliers
        with the exponential smoothing of previous values in the window. The boundaries
        are statically filtered with MAD.
    '''
    if window_size % 2 == 0:
        raise ValueError('Window_size must be an odd number, but value provided is {}'.format(window_size))
    # Create an empty DataFrame to store the cleaned data
    cleaned_df = df.copy()
    
    # Calculate half window size (since the window is centered, it's half to the left and half to the right)
    half_window = (window_size - 1) // 2

    first_part = cleaned_df.loc[:half_window-1,:].to_numpy()
    latent_median = np.median(first_part, axis=0)
    latent_mad = np.median(np.absolute(first_part - latent_median), axis=0)
    mask = np.absolute(first_part - latent_median) > thresh*latent_mad

    for c in range(first_part.shape[1]):
        first_part[mask[:,c],c] = latent_median[c]

    cleaned_df.loc[:half_window-1,:] = first_part

    last_part = cleaned_df.loc[len(cleaned_df)-half_window:,:].to_numpy()
    latent_median = np.median(last_part, axis=0)
    latent_mad = np.median(np.absolute(last_part - latent_median), axis=0)
    mask = np.absolute(last_part - latent_median) > thresh*latent_mad

    for c in range(last_part.shape[1]):
        last_part[mask[:,c],c] = latent_median[c]
    
    cleaned_df.loc[len(cleaned_df)-half_window:,:] = last_part

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

import pywt
import pywt.data
def DWT_denoising(df, wavelet = 'haar', level = 5, threshold = 0.7):
    ''' Performs Discrete Wavelet Transform the an dataframe where eache column is a series
        to smooth out noise and preserve signal information
    '''

    def soft_threshold(value, threshold):
        return np.where(value > threshold, value - threshold, np.where(value < -threshold, value + threshold, 0.0))
    
    coeffs = pywt.wavedec(df, wavelet, level=level, axis=0)
    denoised_coeffs = [soft_threshold(coeff, threshold) for coeff in coeffs]
    df_denoised = pywt.waverec(denoised_coeffs, wavelet, axis=0)
    #this is done because if the original data has an even number of rows, the DWT add an extra row at the end, don't ask me precisely why :)
    if (df.shape[0] % 2) == 1:
        df_denoised = df_denoised[:-1]

    return pd.DataFrame(df_denoised, columns=df.columns)

def create_tensor(df, winsize = 200, overlap =100):
    ''' Create a tensor of images dividing the sequence (dataframe) in windows that overlaps
        according to the parameters
    '''
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

# IMPORTAN!!! From now on these functions are NOT used at all in the wi-fi person ID project, they were tested and refused

from scipy.stats import skew
def add_statistics(input_df):

    # Create DataFrames for each statistic
    mean_row = pd.DataFrame(input_df.mean()).T
    variance_row = pd.DataFrame(input_df.var()).T
    std_row = pd.DataFrame(input_df.std()).T
    median_row = pd.DataFrame(input_df.median()).T
    skewness_row = pd.DataFrame(input_df.apply(lambda x: skew(x, nan_policy='omit'))).T

    # Concatenate the statistics DataFrames to the input DataFrame
    result_df = pd.concat([input_df, mean_row, variance_row , std_row, median_row, skewness_row], ignore_index=True)

    mean_col = result_df.mean(axis=1)
    variance_col = result_df.var(axis=1)
    std_col = result_df.std(axis=1)
    median_col = result_df.median(axis=1)
    skewness_col = result_df.apply(lambda x: skew(x, nan_policy='omit'), axis=1)

    #result_df['Mean'] = mean_col
    result_df['Variance'] = variance_col
    result_df['Std'] = std_col
    result_df['Median'] = median_col
    result_df['Skewness'] = skewness_col

    return result_df

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

from scipy.signal import butter, filtfilt
def butterworth_filter(data, order=5, cutoff_frequency=10, sampling_rate=100):
    """
    Apply a Butterworth filter to each column of the input DataFrame.
    
    Parameters:
    - data (pd.DataFrame): Input DataFrame where each column is a time series.
    - order (int): Order of the Butterworth filter. Default is 5.
    - cutoff_frequency (float): Cutoff frequency of the filter in Hz. Default is 10 Hz.
    - sampling_rate (float): Sampling rate of the time series in Hz. Default is 100 Hz.
    
    Returns:
    - pd.DataFrame: DataFrame with filtered time series.
    """
    
    # Define the Butterworth filter
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter to each column
    filtered_data = data.apply(lambda col: filtfilt(b, a, col), axis=0)
    
    return filtered_data
