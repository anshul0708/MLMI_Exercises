'''Imports CIFAR-10 data.'''
import os
import numpy as np
import pickle
import sys

IMG_SIZE = 32

def load_CIFAR10_batch(filename):
    '''load data from single CIFAR-10 file'''

    with open(filename, 'rb') as f:
        if sys.version_info[0] < 3:
            dict = pickle.load(f)
        else:
            dict = pickle.load(f, encoding='latin1')
        x = dict['data']
        y = dict['labels']
        x = x.astype(float)
        y = np.array(y)
    return x, y


def load_data():
    '''load all CIFAR-10 data and merge training batches'''
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    xs = []
    ys = []
    for i in range(1, 6):
        filename = cur_dir + '/cifar-10-batches-py/data_batch_' + str(i)
        X, Y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(Y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    del xs, ys

    x_test, y_test = load_CIFAR10_batch(
        cur_dir + '/cifar-10-batches-py/test_batch')

    # classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    #   'ship', 'truck']

    classes = ['notcat', 'cat']

    # # Using only cats and dogs
    # ind = np.where(np.logical_or(y_train == 3, y_train == 5))
    
    # y_train = y_train[ind]
    # x_train = x_train[ind, :]
    # x_train = np.reshape(x_train, (x_train.shape[1], x_train.shape[2]))
    # # x_train = np.reshape(x_train, (x_train.shape[0], IMG_SIZE, IMG_SIZE))

    # ind = np.where(np.logical_or(y_test == 3, y_test == 5))
    # y_test = y_test[ind]
    # x_test = x_test[ind, :]
    # x_test = np.reshape(x_test, (x_test.shape[1], x_test.shape[2]))

    ind_cat = np.where(y_train == 3)
    ind_ncat = np.where(y_train != 3)

    ind_cat =ind_cat[0]
    ind_ncat=ind_ncat[0]
    print ind_cat.shape
    ind_ncat = ind_ncat[:int(len(ind_ncat)/9)]
    ind = np.concatenate((ind_cat, np.array(ind_ncat)))
    ind.sort()
    y_train = y_train[ind]
    y_train[y_train != 3] = 0
    y_train[y_train == 3] = 1
    x_train = x_train[ind, :]
    # x_train = np.reshape(x_train, (x_train.shape[1], x_train.shape[2]))
    x_train = np.reshape(x_train, (x_train.shape[0], IMG_SIZE, IMG_SIZE, 3))

    # ind_cat = np.where(y_test == 3)
    # ind_ncat = np.where(y_test != 3)

    # ind_cat =ind_cat[0]
    # ind_ncat=ind_ncat[0]
    # print y_train[:15]
    # ind_ncat = ind_ncat[:int(len(ind_ncat)/9)]
    # ind = np.concatenate((ind_cat, np.array(ind_ncat)))
    # ind.sort()
    # y_test = y_test[ind]
    y_test[y_test != 3] = 0
    y_test[y_test == 3] = 1
    # x_test = x_test[ind, :]
    x_test = np.reshape(x_test, (x_test.shape[0], IMG_SIZE, IMG_SIZE, 3))
    # x_test = np.reshape(x_test, (x_test.shape[0], IMG_SIZE, IMG_SIZE))

    # Normalize Data
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image

    data_dict = {
        'images_train': x_train,
        'labels_train': y_train,
        'images_test': x_test,
        'labels_test': y_test,
        'classes': classes
    }
    return data_dict


def reshape_data(data_dict):
    im_tr = np.array(data_dict['images_train'])
    im_tr = np.reshape(im_tr, (-1, 3, 32, 32))
    im_tr = np.transpose(im_tr, (0, 2, 3, 1))
    data_dict['images_train'] = im_tr
    im_te = np.array(data_dict['images_test'])
    im_te = np.reshape(im_te, (-1, 3, 32, 32))
    im_te = np.transpose(im_te, (0, 2, 3, 1))
    data_dict['images_test'] = im_te
    return data_dict

# def generate_random_batch(images, labels, batch_size):
    # Generate batch
    #indices = np.random.choice(images.shape[0], batch_size)
    #images_batch = images[indices]
    #labels_batch = labels[indices]
    # return images_batch, labels_batch


def gen_batch(data, batch_size, num_iter):
    data = np.array(data)
    index = len(data)
    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > len(data)):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(len(data)))
            data = data[shuffled_indices]
        yield data[index:index + batch_size]


def main():
    data_sets = load_data()
    print(data_sets['images_train'].shape)
    print(data_sets['labels_train'].shape)
    print(data_sets['images_test'].shape)
    print(data_sets['labels_test'].shape)


if __name__ == '__main__':
    main()
