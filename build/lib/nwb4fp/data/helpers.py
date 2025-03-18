import numpy as np
from scipy.ndimage import gaussian_filter
import seaborn as sns
from itertools import chain
import pandas as pd

def df2results(df):
    # Assuming your data is in arrays x, y, and speed
    # If they're in a pandas DataFrame called df, you could extract them like:
    # x = df['x'].values
    # y = df['y'].values
    # speed = df['speed'].values

    # Create 2D histogram with 200x200 bins
    bins = 200
    x_bins = np.linspace(0, 1, bins + 1)  # +1 because these are bin edges
    y_bins = np.linspace(0, 1, bins + 1)

    x_control = list(chain.from_iterable(df['x'])) 
    y_control = list(chain.from_iterable(df['y'])) 
    speeds = list(chain.from_iterable(df['smoothed_speed'])) 


    # Calculate the 2D histogram of counts
    hist_counts, x_edges, y_edges = np.histogram2d(x_control, y_control, bins=[x_bins, y_bins])

    # Calculate the sum of speeds in each bin
    hist_speed_sum, _, _ = np.histogram2d(x_control, y_control, bins=[x_bins, y_bins], weights=speeds)

    # Avoid division by zero by using where
    average_speed = np.divide(hist_speed_sum, hist_counts, 
                            out=np.zeros_like(hist_speed_sum), 
                            where=hist_counts != 0)

    # Get bin centers for x and y (instead of edges)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Now you have:
    # X: 200x200 array of x coordinates
    # Y: 200x200 array of y coordinates
    # average_speed: 200x200 array of averaged speeds
    smoothed_speed = gaussian_filter(average_speed, sigma=20/2) 
    data_speedss = pd.DataFrame(smoothed_speed, index=y_centers, columns=x_centers)

    return data_speedss



def df2results_sns(df):
    x_control = list(chain.from_iterable(df['x'])) 
    y_control = list(chain.from_iterable(df['y'])) 


    sample_size = min(100000, len(x_control))  # Don’t exceed original size
    indices = np.random.choice(len(x_control), sample_size, replace=False)

    x_sampled = [x_control[i] for i in indices]
    y_sampled = [y_control[i] for i in indices]

    # Create sampled DataFrame
    data = pd.DataFrame({'x': x_sampled, 'y': y_sampled})
    return data