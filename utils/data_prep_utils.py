import os
import cv2
import glob
import scipy.io
import scipy.misc
from tqdm import tqdm

DATES = ['24032019', '17042019', '01052019']
LABELS = ['yafim', 'zeev', 'or', 'ron', 'sergey', 'aviya', 'elnatan', 'felix']


def prep_data(dates, labels, frames_size=32):
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


def show_data_example(example_path='../data/videos/24032019/yafim/32/VideoFile_fps100_0002.mat',
                      example_key='Video_fps100_0002'):
    mat = scipy.io.loadmat(example_path)
    img_reshaped = mat[example_key][:, :]
    print('example shape: ', img_reshaped.shape)
    print('example keys:')
    for k, v in mat.items():
        print(k, v)
    for i in range(img_reshaped.shape[2]):
        cv2.imshow("segmented_map", mat['Video_fps100_0002'][:, :, i])
        cv2.waitKey(0)
