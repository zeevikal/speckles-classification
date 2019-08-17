import os
import cv2
import glob
import scipy.io
import scipy.misc
from tqdm import tqdm

# Data collection dates
DATES = ['24032019', '17042019', '01052019']
# Data labels
LABELS = ['yafim', 'zeev', 'or', 'ron', 'sergey', 'aviya', 'elnatan', 'felix']
# Frames images size
FRAMES_SIZE = ['32', '64', '128']
# Data categories (according to 'Photonic Human Identification based on Deep Learning
# of Back Scattered Laser Speckle Patterns' paper).
VID_CAT = ['regular', 'angry', 'smile', 'brows_up', 'sad', 'water', 'sport']


def prep_data(dates, labels, frames_size=32):
    """
    This function preparing the data according to data paths and structure.
    :param dates: Data collection dates
    :param labels: Data labels
    :param frames_size: Frames images size (default is 32x32)
    :return:
    """
    for d in dates:
        for l in labels:
            for f in tqdm(glob.glob('../data/videos/{0}/{1}/{2}/*.mat'.format(d, l, frames_size))):
                frames_path = '../data/frames/{0}/{1}/{2}'.format(d, l, frames_size)
                if not os.path.exists(frames_path):
                    os.makedirs(frames_path)
                mat = scipy.io.loadmat(f)
                for k, v in mat.items():
                    if k.__contains__('Video_fps'):
                        for i in range(v.shape[2]):
                            cv2.imwrite('{0}/{1}_{2}.png'.format(frames_path, k, i), v[:, :, i])


def prep_categorized_data(dates, labels, vid_cat, frames_size=32):
    """
    This function preparing the data according to data paths, structure
    and categories (according to 'Photonic Human Identification based on Deep Learning
    of Back Scattered Laser Speckle Patterns' paper).
    :param dates: Data collection dates
    :param labels: Data labels
    :param vid_cat: Data categories
    :param frames_size: Frames images size (default is 32x32)
    :return:
    """
    for d in dates:
        for l in labels:
            for vc in vid_cat:
                for f in tqdm(
                        glob.glob('../data/categorized-video/{0}/{1}/{2}/{3}/*.mat'.format(d, l, frames_size, vc))):
                    frames_path = '../data/categorized_frames/{0}/{1}/{2}/{3}'.format(d, l, frames_size, vc)
                    if not os.path.exists(frames_path):
                        os.makedirs(frames_path)
                    mat = scipy.io.loadmat(f)
                    for k, v in mat.items():
                        if k.__contains__('Video_fps'):
                            for i in range(v.shape[2]):
                                cv2.imwrite('{0}/{1}_{2}.png'.format(frames_path, k, i), v[:, :, i])


def show_data_example(example_path='../data/videos/24032019/yafim/32/VideoFile_fps100_0002.mat',
                      example_key='Video_fps100_0002'):
    """
    Pring and Plot data (frame) example according to specific data video.
    :param example_path: video file path (.mat file)
    :param example_key: .mat file dict key
    :return:
    """
    mat = scipy.io.loadmat(example_path)
    img_reshaped = mat[example_key][:, :]
    print('example shape: ', img_reshaped.shape)
    print('example keys:')
    for k, v in mat.items():
        print(k, v)
    for i in range(img_reshaped.shape[2]):
        cv2.imshow("segmented_map", mat[example_key][:, :, i])
        cv2.waitKey(0)
