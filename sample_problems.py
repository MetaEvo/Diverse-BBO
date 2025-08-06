import datetime
import pickle
from net.AE import load_model, get_encoded
import numpy as np
import matplotlib.pyplot as plt
import os

from train_AE import generate_dataset, normalize_data
from utils import set_seed

def sample_points_in_rectangle(points, N, save_path):
    nowTime=datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    save_path = save_path+'/'+nowTime
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Calculate the rectangle boundaries
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
    
    # Calculate the width and height of the rectangle
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate the number of points along each dimension
    aspect_ratio = width / height
    num_cols = int(np.sqrt(N * aspect_ratio))  # Number of points along x-axis
    num_rows = int(N / num_cols)              # Number of points along y-axis
    
    # Adjust if necessary
    if num_cols * num_rows < N:
        num_cols += 1

    # Create grid points within the rectangle
    x_points = np.linspace(min_x, max_x, num_cols)
    y_points = np.linspace(min_y, max_y, num_rows)
    
    # Generate all combinations of (x, y) within the grid
    grid_points = np.array([(x, y) for x in x_points for y in y_points])

    selected_points_3d = grid_points.reshape(num_rows, num_cols, 2)
    
    # Save the 3D array as .npy file
    np.save(os.path.join(save_path,f'sample_points.npy'), selected_points_3d)
    return selected_points_3d

if __name__ =='__main__':
    set_seed(100)
    dim = 10
    save_bbob_ela = './ela_store'
    model_path = './models/autoencoder_epoch_300.pth'
    with open(os.path.join(save_bbob_ela,f'example_mixedDimension_BBOB_ela_.pickle'), 'rb') as f:
        dataset = pickle.load(f)
    n_fea = dataset.shape[-1] 
    #min max
    normalized_data, scaler = normalize_data(dataset)
    normalized_data = normalized_data.reshape(dataset.shape[0],n_fea)

    model = load_model(model_path,n_fea)
    points = get_encoded(model,normalized_data)
    N = 120  # nums of sample points
    save_path = "./sample_points"
    # print(points.shape)
    sampled_points_3d = sample_points_in_rectangle(points.reshape(-1,2),N, save_path)
    # print(sampled_points_3d.shape)
