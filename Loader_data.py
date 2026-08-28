import scipy.io as sio
import numpy as np
import os


def load_data(dataset_name):
    X = []
    y = []

    if dataset_name == 'mirage':

        path = './Vector_data/mirage_news_'
        path_img = ['img_train', 'img_valid', 'img_test']
        path_text = ['text_train', 'text_valid', 'text_test']

        for i in range(3):
            path_imge = path + path_img[i]
            path_text_ = path + path_text[i]
            data_img = sio.loadmat(path_imge)
            data_text = sio.loadmat(path_text_)
            X.append([data_img['X'], data_text['X']])
            y.append(data_img['Y'])

    return X, y


if __name__ == '__main__':
    X, y = load_data('mirage')
    print(X[0][0][0].shape)

