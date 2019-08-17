import argparse
import json
import sys

from utils.train_utils import *


def parse_json(json_path):
    """
    Parse training params json file to python dictionary
    :param json_path: path to training params json file
    :return: python dict
    """
    with open(json_path) as f:
        d = json.load(f)
        return d


def start_train(train_dict):
    """
    Starting the training  process according to training params dictionary.
    :param train_dict: training params dictionary
    :return:
    """
    x_train, x_valid, x_test, y_train, y_valid, y_test = prep_data_train_val_test(
        dates=train_dict['dates'],
        labels=train_dict['labels'],
        data_path=train_dict['data_path'],
        is_dim_3d=train_dict['is_dim_3d'])

    run_custom_training(conf_list=train_dict['conf_list'],
                        labels=train_dict['labels'],
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_valid,
                        y_test=y_valid,
                        model_path=train_dict['model_path'],
                        model_path_prefix=train_dict['model_path_prefix'],
                        n_batch_size=train_dict['n_batch_size'],
                        n_epochs=train_dict['n_epochs'])


def start_test(test_dict):
    """
    Starting the testing process according to training params dictionary.
    :param test_dict:
    :return:
    """
    x_train, x_valid, x_test, y_train, y_valid, y_test = prep_data_train_val_test(
        dates=test_dict['dates'],
        labels=test_dict['labels'],
        data_path=test_dict['data_path'],
        is_dim_3d=test_dict['is_dim_3d'])
    test_model(pre_trained_model_path=test_dict['pre_trained_model_path'],
               labels=test_dict['labels'],
               dates=test_dict['dates'],
               categorized_frames_path=test_dict['categorized_frames_path'],
               frame_size=test_dict['frame_size'],
               x_test=x_test,
               y_test=y_test,
               to_plot=test_dict['to_plot'])


def main():
    parser = argparse.ArgumentParser(description='Speckles classification tarining process')
    parser.add_argument('-m', '--mode', help='process mode (train or valid)', required=True,
                        default='train')
    parser.add_argument('-j', '--json_path', help='Json path with process details', required=True,
                        default='training_params.json')
    args = vars(parser.parse_args())

    if args['mode'].lower() == 'train':
        d = parse_json(args['json_path'])['train']
        d['is_dnn'] = True if d['is_dnn'].lower() == "true" else False
        d['is_dim_3d'] = True if d['is_dim_3d'].lower() == "true" else False
        start_train(d)

    elif args['mode'].lower() == 'test':
        d = parse_json(args['json_path'])['test']
        d['to_plot'] = True if d['to_plot'].lower() == "true" else False
        d['is_dim_3d'] = True if d['is_dim_3d'].lower() == "true" else False
        start_test(d)
    else:
        sys.exit("Wrong mode! please choose between: 'train' and 'test'")


if __name__ == '__main__':
    main()
