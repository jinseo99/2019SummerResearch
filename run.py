"""
By: Jinseo Lee

This file shows the sample run of different heatmap algorithms 
"""

import sys
import os 
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path +'/HeatMapAlgorithms') 
sys.path.append(dir_path +'/utility') 

from CAM import CAM_1 
from load_model import loadModel
from load_data import load_data

if __name__ == "__main__":
    filename0 = "some_model_name"
    model = loadModel(filename0)
    last_conv_layer = -1 # you can get this information from model layers
    filename1 = "some_data_file"
    filename2 = "some_data_category_file"

    x, y = load_data(filename1, filename2)
    heatmap = CAM_1(x[0], last_conv_layer, model)
    plt.imshow(heatmap, cmap = 'jet')