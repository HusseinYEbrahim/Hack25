import sys
sys.path.append('../')

import requests
import numpy as np
from LSBSteg import decode

import tensorflow as tf
import time 
import json

model_path = '../Eagle/saved_eagle'

model = tf.saved_model.load(model_path)

infer = model.signatures["serving_default"]

file_path = './jsonnn.txt'

# Read the file and parse its content to a JSON object
with open(file_path, 'r') as file:
    json_object = json.load(file)

print(json['1'].shape)

def get_model_prediction(data):
    input_data = tf.constant(data.reshape((1, 1998, 101, 1)), dtype=tf.float32)
    result = infer(input_data)  
    result_value = result['dense_1'].numpy()
    if(result_value[0][0] > 0.5):
        return 'R'
    return 'F'

def remove_nans(data):
    spectrograms = data['x']
    inf_indices = np.isinf(spectrograms)

    # Replace 'inf' values with NaN
    spectrograms[inf_indices] = np.nan

    # Iterate over each sample and replace 'inf' values with the mean along axis=1
    for i in range(spectrograms.shape[0]):
        sample = spectrograms[i, :, :]
        mean_values = np.nanmean(sample, axis=0, keepdims=True)
        mean_values[np.isinf(mean_values)] = 0
        inf_sample_indices = np.isnan(sample)
        sample[inf_sample_indices] = np.tile(mean_values, (sample.shape[0], 1))[inf_sample_indices]
        spectrograms[i, :, :] = sample
    
    return spectrograms