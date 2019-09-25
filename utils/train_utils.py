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

from utils.data_prep_utils import LABELS, VID_CAT

np.random.seed(7)

# configuration list for grid search (DNN & CNN)
DNN_CONF_LIST = [
    [32, 256, 256, 0.2],
    [32, 512, 512, 0.1],
    [32, 512, 512, 0.2],
    [32, 300, 100, 0.2],
    [32, 300, 100, 0.4],
    [32, 512, 256, 0.4],
    [32, 512, 256, 0.2],
    [32, 256, 128, 0.4],
    [32, 256, 128, 0.2]
]
CNN_CONF_LIST = [
    [32, 64, 3, 2, 0.4, 256, 0.4, 128, 0.4],
    [32, 32, 3, 2, 0.3, 256, 0.3, 128, 0.3],
    [32, 64, 5, 2, 0.4, 128, 0.4, 64, 0.4],
    [32, 64, 5, 2, 0.4, 256, 0.4, 128, 0.4],
    [32, 128, 3, 2, 0.4, 256, 0.4, 128, 0.4],
    [32, 32, 3, 2, 0.2, 256, 0.2, 128, 0.2]
]

# training params
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


def plt_show():
    try:
        plt.show()
    except UnicodeDecodeError:
        plt_show()


def prep_data_train_test(dates, labels, x_data=[], y_data=[]):
    """
    Prepare train and test datasets for training process,
    according to the data structure.
    :param dates: The dates on which the data was collected
    :param labels: Training labels
    :param x_data: Training X data
    :param y_data: Training Y data
    :return: train and test datasets -  x_train, x_test, y_train, y_test
    """
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
                             valid_precent_split=VALID_PERCENT_SPLIT, is_cnn=False):
    """
    Prepare train, validation and test datasets for training process,
    according to the data structure.
    :param dates: The dates on which the data was collected
    :param labels: Training labels
    :param data_path: Path to our data (frames data)
    :param frames_size: frame size param (32, 64 or 128)
    :param valid_and_test: [training] & [val, test] data split ration
    :param valid_precent_split: [val] & [test] data split ration
    :param is_cnn: True if frames data are for CNN architecture, else (for DNN) False
    :return: train, validation and test datasets - x_train, x_valid, x_test, y_train, y_valid, y_test
    """
    x_train = []
    y_train = []

    x_valid = []
    y_valid = []

    x_test = []
    y_test = []

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

    if is_cnn:
        x_train = x_train.reshape((-1, frames_size, frames_size, 1))
        x_valid = x_valid.reshape((-1, frames_size, frames_size, 1))
        x_test = x_test.reshape((-1, frames_size, frames_size, 1))

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def show_sample(x_data, idx):
    """
    Show data sample. for example: show_sample(x_train, 5),
    where x_train is data that contains frames and 5 is an index.
    :param x_data: data
    :param idx: index (in this specific data)
    :return:
    """
    plt.imshow(x_data[idx])
    plt_show()


def print_unique_data(y_test):
    """
    Count and print unique data (as labels).
    :param y_test: data to count and print
    :return:
    """
    unique, counts = np.unique(y_test, return_counts=True)
    print(np.asarray((unique, counts)).T)


def set_curr_time():
    """
    set current timestamp for documenting models names.
    :return:
    """
    dt = datetime.datetime.now()
    curr_dt = '{0}{1}{2}_{3}_{4}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    return curr_dt


def custom_dnn_model(conf, labels, model_weights=None):
    """
    Deep Neural network architecture based on 'Photonic Human Identification based on
    Deep Learning of Back Scattered Laser Speckle Patterns' paper.
    :param conf: Configuration list of models hyper & learning params
    :param labels: List of data labels
    :param model_weights: Weights of pre-trained model
    :return: DNN model
    """
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


def custom_cnn_model(config, labels, model_weights=None):
    """
    Convolutional Neural network architecture based on 'Photonic Human Identification based on
    Deep Learning of Back Scattered Laser Speckle Patterns' paper.
    :param conf: Configuration list of models hyper & learning params
    :param labels: List of data labels
    :param model_weights: Weights of pre-trained model
    :return: CNN model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(config[1], config[2], input_shape=(config[0], config[0], 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(config[3], config[3])),
        tf.keras.layers.Dropout(config[4]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(config[5]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.nn.relu),
        tf.keras.layers.Dropout(config[6]),
        tf.keras.layers.Dense(config[7]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.nn.relu),
        tf.keras.layers.Dropout(config[8]),
        tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)
    ])
    if model_weights is not None:
        print('loading pre-trained model')
        model.load_weights(model_weights, by_name=True)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_custom_training(conf_list, labels, x_train, y_train, x_test, y_test, n_epochs=50, n_batch_size=256,
                        model_path='../models', is_dnn=True, model_path_prefix=None, model_weights=None):
    """
    A training process function. this function loads model according to its type
    and its unique configuration file.
    Tensorboard callback is enabled and a log file saved according to the configuration file.
    Finally, the model is saved according to its name and a given path.
    :param conf_list: model params configuration list
    :param labels: List of data labels
    :param x_train: training set data
    :param y_train: training set labels
    :param x_test: validation set data
    :param y_test: validation set labels
    :param n_epochs: number of training epochs
    :param n_batch_size: batch size
    :param model_path: path to save model file
    :param is_dnn: if True a DNN custom function will be called, else CNN function will be called
    :param model_path_prefix: Save model file name prefix
    :param model_weights: Path to pre-trained model. if None, initialize models weighs from scratch.
    :return:
    """
    for conf in conf_list:
        curr_dt = set_curr_time()
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/{0}_{1}'.format(conf, curr_dt),
                                                     histogram_freq=0, write_graph=True)
        if is_dnn:
            model = custom_dnn_model(conf, labels, model_weights)
        else:
            model = custom_cnn_model(conf, labels, model_weights)
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=n_batch_size,
                  epochs=n_epochs, callbacks=[tb_callback])
        model.evaluate(x_test, y_test)
        if model_path_prefix is None:
            model.save('{0}/{1}_{2}.h5'.format(model_path, conf, curr_dt))
        else:
            model.save('{0}/{1}_{2}_{3}.h5'.format(model_path, model_path_prefix, conf, curr_dt))


def prep_confusion_matrix(y_test, test_predictions, labels, is_return=False):
    """
    Prepare and plot confusion matrix depending on
    the test set of results (predictions).
    :param y_test: test set labels
    :param test_predictions: models predictions list
    :param labels: List of data labels
    :param is_return: if True, we'll return the Confusion Matrix DataFrame
    :return: Confusion Matrix DataFrame (if is_return is True)
    """
    confusion = confusion_matrix(y_test, np.argmax(test_predictions, axis=1))
    df_cm = pd.DataFrame(confusion, labels, labels)
    plt.figure(figsize=(12, 9))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, fmt='.5g', annot_kws={"size": 12})
    if is_return:
        return df_cm


def prep_predicted_dict(labels, dates, categorized_frames_path='../data/categorized_frames',
                        frame_size=32):
    """
    Prepare redicted dictionary
    :param labels: List of data labels
    :param dates: The dates on which the data was collected
    :param categorized_frames_path: categorized frames data path
    :param frame_size: frame size
    :return:
    """
    dd = {}
    for i, l in enumerate(labels):
        dd[l] = {}
        for vc in VID_CAT:
            dd[l][vc] = {}
            dd[l][vc]['x'] = []
            dd[l][vc]['y'] = []
            for d in dates:
                for f in tqdm(glob.glob('{0}/{1}/{2}/{3}/{4}/*.png'.format(categorized_frames_path,
                                                                           d, l, frame_size, vc))):
                    dd[l][vc]['x'].append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
                    dd[l][vc]['y'].append(i)
            dd[l][vc]['x'] = np.asarray(dd[l][vc]['x'])
            dd[l][vc]['y'] = np.asarray(dd[l][vc]['y'])
    return dd


def plot_per_category(labels, dd, loaded_model):
    """
    Plot 1st graph according to our paper
    :param labels: List of data labels
    :param dd: Data dictionary for plotting
    :param loaded_model: trained loaded model
    :return:
    """
    for k, v in tqdm(dd.items()):
        for kk, vv in v.items():
            if len(vv['x']) > 0:
                vv['predicted'] = loaded_model.predict(vv['x'])
    for i, l in enumerate(labels):
        auto_corr = []
        var_by_category = []
        for vc in VID_CAT:
            auto = 0
            var_list = []
            for p in dd[l][vc]['predicted']:
                auto += p[i]
                var_list.append(p[i])
            #             print(l, vc, auto / len(dd[l][vc]['predicted']) * 100)
            auto_corr.append(auto / len(dd[l][vc]['predicted']))
            var_by_category.append(np.var(var_list))

        x_pos = [i for i, _ in enumerate(VID_CAT)]
        plt.bar(x_pos, auto_corr, color='green', yerr=var_by_category)
        plt.xlabel("categories")
        plt.ylabel("accuracy")
        plt.title("{0} categories & accuracy".format(l))
        plt.xticks(x_pos, VID_CAT)
        plt_show()


def plot_one_vs_all(labels, dd, loaded_model):
    """
    Plot 2nd graph according to our paper
    :param labels: List of data labels
    :param dd: Data dictionary for plotting
    :param loaded_model: trained loaded model
    :return:
    """
    for k, v in tqdm(dd.items()):
        for kk, vv in v.items():
            if len(vv['x']) > 0:
                vv['predicted'] = loaded_model.predict(vv['x'])
    for i, l in enumerate(labels):

        auto_var_list = []
        cross_var_list = []

        auto = 0
        cross = 0

        l_len = 0

        for vc in VID_CAT:
            for p in dd[l][vc]['predicted']:
                auto += p[i]
                l_len += 1
                for idx, score in enumerate(p):
                    if idx is not i:
                        cross += p[idx]
                auto_var_list.append(p[i])
                cross_var_list.append(np.mean(cross / len(dd[l][vc]['predicted'])))

        x_pos = [0, 1]
        total_corr = [(auto / l_len), (cross / l_len)]
        total_var = [(np.var(auto_var_list)), (np.var(cross_var_list))]
        plt.bar(x_pos, total_corr, color='green', yerr=total_var)
        plt.ylabel("accuracy")
        plt.title("auto & cross prediction for {0}".format(l))
        plt.xticks(x_pos, ['auto', 'cross'])
        plt_show()


def test_model(pre_trained_model_path, labels, dates, categorized_frames_path,
               frame_size, x_test, y_test, to_plot=True):
    """
    run test process on trained model with or without plotting
    :param pre_trained_model_path: pre-trained model path
    :param labels: List of data labels
    :param dates: The dates on which the data was collected
    :param categorized_frames_path: categorized frames data path
    :param frame_size: frame size
    :param x_test: validation set data
    :param y_test: validation set labels
    :param to_plot: if True, will plot graphs (according to our paper)
    :return:
    """
    loaded_model = tf.keras.models.load_model(pre_trained_model_path)
    print(loaded_model.summary())
    test_predictions = loaded_model.predict(x_test)
    print('{0} results on your test set:'.format(pre_trained_model_path))
    print(loaded_model.evaluate(x_test, y_test))
    if to_plot:
        predicted_dict = prep_predicted_dict(labels, dates, categorized_frames_path, frame_size)
        plot_per_category(labels, predicted_dict, loaded_model)
        plot_one_vs_all(labels, predicted_dict, loaded_model)
