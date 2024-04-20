import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import load_model
import cv2 
import mlflow
from utils import image_to_pixels, preprocess_pixels

import re

model = load_model('./models/cnn_6.h5')
pattern = r"sudoku-*"
rel_path = './images/toy/'
my_digits = [filename for filename in os.listdir(rel_path) if re.match (pattern, filename)]

for image_file in my_digits:
    image = cv2.imread(rel_path + image_file)
    image = image_to_pixels(image)
    image = preprocess_pixels(image)

    prediction = model.predict(image)
    prediction_label = prediction.argmax() + 1
    predicted_probability = prediction.max()

    print(f'{image_file = }')
    print(f'{prediction.round(3) = }')
    print(f'{prediction_label = }')
    print(f'{predicted_probability = }')
    print()
    print()
    print()