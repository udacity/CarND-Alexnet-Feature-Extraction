import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: load data

# TODO: split data into training and validation sets

# TODO: define placeholders and resize operation

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the traffic signs
# model.
fc7 = AlexNet(..., feature_extract=True)

# TODO: add the final layer for traffic sign classification

# TODO: define loss, training, accuracy operations
# HINT: look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: train and evaluate the feature extraction model
