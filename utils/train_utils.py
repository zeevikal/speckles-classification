from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import seaborn as sn
import pandas as pd
import numpy as np
import itertools
import datetime
import glob
import cv2

from utils.data_prep_utils import LABELS

np.random.seed(7)

CONF_LIST = [
    [32, 512, 0.2],
    [32, 256, 0.2],
    [32, 512, 0.1],
]

N_INPUTS = 32 * 32
N_HIDDEN1 = 300
N_HIDDEN2 = 100

N_OUTPUTS = len(LABELS)
BATCH_NORM_MOMENTUM = 0.9
LR = 0.001

TRAIN_PERCENT_SPLIT = 0.8
VALID_PERCENT_SPLIT = 0.5
VALID_AND_TEST = 1 - TRAIN_PERCENT_SPLIT

N_EPOCHS = 15
BATCH_SIZE = 256

model_path_prefix = 'tf_core'


def prep_data_train_test(dates, labels, x_data=[], y_data=[]):
    for d in dates:
        for i, l in enumerate(labels):
            for f in tqdm(glob.glob('../data/frames/{0}/{1}/32/*.png'.format(d, l))):
                x_data.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
                y_data.append(i)

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, x_test, y_train, y_test


def prep_data_train_val_test(dates, labels, data_path='../data/frames', frames_size=32, valid_and_test=VALID_AND_TEST,
                             valid_precent_split=VALID_PERCENT_SPLIT, is_dim_3d=False):
    x_train = []
    y_train = []

    x_valid = []
    y_valid = []

    x_test = []
    y_test = []

    # for i, l in enumerate(labels):
    for d in dates:
        for i, l in enumerate(labels):
            x_data = []
            y_data = []
            for f in tqdm(glob.glob('{0}/{1}/{2}/{3}/*.png'.format(data_path, d, l, frames_size))):
                x_data.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
                y_data.append(i)
            x_train_label, xx_test_label, y_train_label, yy_test_label \
                = train_test_split(x_data, y_data, test_size=valid_and_test)
            x_valid_label, x_test_label, y_valid_label, y_test_label \
                = train_test_split(xx_test_label, yy_test_label, test_size=valid_precent_split)

            x_train.append(x_train_label)
            y_train.append(y_train_label)
            x_test.append(x_test_label)
            y_test.append(y_test_label)
            x_valid.append(x_valid_label)
            y_valid.append(y_valid_label)

    x_train = list(itertools.chain.from_iterable(x_train))
    y_train = list(itertools.chain.from_iterable(y_train))
    x_test = list(itertools.chain.from_iterable(x_test))
    y_test = list(itertools.chain.from_iterable(y_test))
    x_valid = list(itertools.chain.from_iterable(x_valid))
    y_valid = list(itertools.chain.from_iterable(y_valid))

    x_valid = np.asarray(x_valid)
    y_valid = np.asarray(y_valid)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    if is_dim_3d:
        x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
        x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])
        x_valid = np.reshape(x_valid, [x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1])

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def show_sample(x_data, idx):
    plt.imshow(x_data[idx])
    plt.show()


def print_unique_data(y_test):
    unique, counts = np.unique(y_test, return_counts=True)
    print(np.asarray((unique, counts)).T)


def set_curr_time():
    dt = datetime.datetime.now()
    curr_dt = '{0}{1}{2}_{3}_{4}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    return curr_dt


def custom_model(conf, labels, model_weights=None):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(conf[0], conf[0])),
        tf.keras.layers.Dense(conf[1], activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(conf[2], activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(conf[3]),
        tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)
    ])
    if model_weights is not None:
        print('loading pre-trained model')
        model.load_weights(model_weights, by_name=True)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_custom_training(conf_list, labels,  x_train, y_train, x_test, y_test, n_epochs=50, n_batch_size=256,
                        model_path='../models', model_path_prefix=None, model_weights=None):
    for conf in conf_list:
        curr_dt = set_curr_time()
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/{0}_{1}'.format(conf, curr_dt),
                                                     histogram_freq=0, write_graph=True)
        model = custom_model(conf, labels, model_weights)
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=n_batch_size,
                  epochs=n_epochs, callbacks=[tb_callback])
        model.evaluate(x_test, y_test)
        if model_path_prefix is None:
            model.save('{0}/{1}_{2}.h5'.format(model_path, conf, curr_dt))
        else:
            model.save('{0}/{1}_{2}_{3}.h5'.format(model_path, model_path_prefix, conf, curr_dt))


def prep_confusion_matrix(y_test, test_predictions, labels):
    confusion = confusion_matrix(y_test, np.argmax(test_predictions, axis=1))
    df_cm = pd.DataFrame(confusion, labels, labels)
    plt.figure(figsize=(12, 9))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, fmt='.5g', annot_kws={"size": 12})
    # return df_cm
