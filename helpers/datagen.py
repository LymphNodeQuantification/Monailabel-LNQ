import os
import random
import sys
import threading
import time

import numpy as np
import SimpleITK as sitk
import pandas as pd
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append('..')
from helpers.settings import sheets_folder, images_folder, arrays_folder


class threadsafe_iter:
    # source: https://gist.github.com/renexu/859d05fa3df4509b676fd31bd220ec1b
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def datagen(input_folder, df, batch_factor, augment=True):
    cube_shape = 64
    n_classes = 2
    data_categories = ['positive', 'negative', 'nodes', 'negative']
    batch_size = batch_factor * len(data_categories)
    while True:
        X_batch = np.zeros((batch_size, cube_shape, cube_shape, cube_shape, 1), np.float32)
        y_batch = np.zeros((batch_size, cube_shape, cube_shape, cube_shape, n_classes), np.uint8)
        counter = 0
        for category in data_categories:
            df_f = df[df['data category'] == category]
            random_cases = df_f.sample(n=batch_factor)
            for label in random_cases.label:
                label_path = os.path.join(images_folder, input_folder, category, label)
                label_nda = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
                image_nda = sitk.GetArrayFromImage(sitk.ReadImage(label_path.replace('_label', '')))
                image_nda = np.clip(image_nda, -400, 1000).astype(np.float32)
                image_nda += 400
                image_nda /= 1400
                # print(category, label, image_nda.shape)
                if augment:
                    swaps = np.random.choice([-1, 1], size=(1, 3))
                    txpose = np.random.permutation([0, 1, 2])
                    image_nda = image_nda[::swaps[0, 0], ::swaps[0, 1], ::swaps[0, 2]]
                    image_nda = np.transpose(image_nda, tuple(list(txpose)))
                    label_nda = label_nda[::swaps[0, 0], ::swaps[0, 1], ::swaps[0, 2]]
                    label_nda = np.transpose(label_nda, tuple(list(txpose)))
                label_multi_class = np.zeros((label_nda.shape[0],
                                              label_nda.shape[1],
                                              label_nda.shape[2], n_classes), np.uint8)
                for i in range(n_classes):
                    tmp = np.copy(label_nda)
                    tmp[tmp != i+1] = 255
                    tmp[tmp == i+1] = 1
                    label_multi_class[..., i] = tmp
                label_multi_class[label_multi_class == 255] = 0
                X_batch[counter, ..., 0] = image_nda
                y_batch[counter] = label_multi_class
                counter += 1
        random_indices = np.arange(batch_size)
        np.random.shuffle(random_indices)
        # return X_batch[random_indices], y_batch[random_indices]
        yield X_batch[random_indices], y_batch[random_indices]


@threadsafe_generator
def datagen_timc(input_folder, df, batch_factor, augment=True):
    cube_shape = 64
    n_classes = 2
    data_categories = ['nodes']
    batch_size = batch_factor * len(data_categories)
    while True:
        X_batch = np.zeros((batch_size, cube_shape, cube_shape, cube_shape, 1), np.float32)
        y_batch = np.zeros((batch_size, cube_shape, cube_shape, cube_shape, n_classes), np.uint8)
        counter = 0
        for category in data_categories:
            df_f = df[df['data category'] == category]
            random_cases = df_f.sample(n=batch_factor)
            for label in random_cases.label:
                label_path = os.path.join(images_folder, input_folder, category, label)
                label_nda = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
                image_nda = sitk.GetArrayFromImage(sitk.ReadImage(label_path.replace('_label', '')))
                image_nda = np.clip(image_nda, -400, 1000).astype(np.float32)
                image_nda += 400
                image_nda /= 1400
                # print(category, label, image_nda.shape)
                if augment:
                    swaps = np.random.choice([-1, 1], size=(1, 3))
                    txpose = np.random.permutation([0, 1, 2])
                    image_nda = image_nda[::swaps[0, 0], ::swaps[0, 1], ::swaps[0, 2]]
                    image_nda = np.transpose(image_nda, tuple(list(txpose)))
                    label_nda = label_nda[::swaps[0, 0], ::swaps[0, 1], ::swaps[0, 2]]
                    label_nda = np.transpose(label_nda, tuple(list(txpose)))
                label_multi_class = np.zeros((label_nda.shape[0],
                                              label_nda.shape[1],
                                              label_nda.shape[2], n_classes), np.uint8)
                for i in range(n_classes):
                    tmp = np.copy(label_nda)
                    tmp[tmp != i+1] = 255
                    tmp[tmp == i+1] = 1
                    label_multi_class[..., i] = tmp
                label_multi_class[label_multi_class == 255] = 0
                X_batch[counter, ..., 0] = image_nda
                y_batch[counter] = label_multi_class
                counter += 1
        random_indices = np.arange(batch_size)
        np.random.shuffle(random_indices)
        # return X_batch[random_indices], y_batch[random_indices]
        yield X_batch[random_indices], y_batch[random_indices]


if __name__ == '__main__':

    split_df = pd.read_csv(os.path.join(sheets_folder, 'split', 'cubes_with_folds.csv'))
    input_folder = os.path.join(images_folder, 'ct_lymph_nodes_break_down_64')
    output_folder = os.path.join(arrays_folder, 'tmp')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    start_time = time.time()
    for fold in range(1, 2):
        train_df = split_df[split_df.fold != fold]
        val_df = split_df[split_df.fold == fold]
        train_generator = datagen(input_folder, train_df, 3)
        np.save(os.path.join(output_folder, 'X.npy'), train_generator[0])
        np.save(os.path.join(output_folder, 'y.npy'), train_generator[1])
    print("took {0:.2f} seconds ---".format(time.time() - start_time))
