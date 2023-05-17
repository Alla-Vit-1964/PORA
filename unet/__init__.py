import os

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm_notebook, tqdm
from skimage.draw import ellipse, polygon

os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.losses import binary_crossentropy

from keras import backend as K

from tqdm import tqdm_notebook
import datetime

DIR_DATA = os.path.join(os.getcwd(), 'data', f'{datetime.date.today()}')


def get_DIR_DATA():
    if not os.path.isdir(DIR_DATA):
        os.mkdir(DIR_DATA)
    return DIR_DATA


DIR_IMG = os.path.join(get_DIR_DATA(), 'img')


def get_DIR_IMG():
    if not os.path.isdir(DIR_IMG):
        os.mkdir(DIR_IMG)
    return DIR_IMG


DIR_IMG_RESULT = os.path.join(get_DIR_IMG(), 'result')


def get_DIR_IMG_RESULT():
    if not os.path.isdir(DIR_IMG_RESULT):
        os.mkdir(DIR_IMG_RESULT)
    return DIR_IMG_RESULT


DIR_IMG_MASK = os.path.join(get_DIR_IMG(), 'mask')


def get_DIR_IMG_MASK():
    if not os.path.isdir(DIR_IMG_MASK):
        os.mkdir(DIR_IMG_MASK)
    return DIR_IMG_MASK


DIR_IMG_INPUT = os.path.join(get_DIR_IMG(), 'input')


def get_DIR_IMG_INPUT():
    if not os.path.isdir(DIR_IMG_INPUT):
        os.mkdir(DIR_IMG_INPUT)
    return DIR_IMG_INPUT


DIR_DATA_OBJ = os.path.join(get_DIR_DATA(), 'data_obj')


def get_DIR_DATA_OBJ():
    if not os.path.isdir(DIR_DATA_OBJ):
        os.mkdir(DIR_DATA_OBJ)
    return DIR_DATA_OBJ


def exists_data_obj(name):
    return os.path.exists(os.path.join(get_DIR_DATA_OBJ(), f'{name}.pickle'))
