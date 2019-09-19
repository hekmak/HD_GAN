import numpy as np
import tensorflow as tf
import pathlib
import random
import os
import matplotlib.pyplot as plt
import argparse
AUTOTUNE = tf.data.experimental.AUTOTUNE
from data_parser import data_parser

if __name__=='__main__':
    # for verifying the parsed dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #tf.enable_eager_execution()
    with tf.Session() as sess:
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--dataset_dir', dest='dataset_dir', default='/notebooks/dataset/GANs/Persian_miniature', help='path of the dataset')
        parser.add_argument('--max_depth', dest='max_depth', type=float, default=1, help='maximum depth')
        args = parser.parse_args()
        ds = data_parser(args=args)
        ds_train = ds.create_dataset()

        ds = ds_train.shuffle(buffer_size=1000)
        ds = ds.repeat()

        #ds = ds.batch(1)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(1))
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        # creating the training iterator
        iter = ds.make_one_shot_iterator()

        rgb = iter.get_next()

        image_mean, image_var = tf.nn.moments(rgb,axes=[0,1,2])

        #sess = tf.Session()
        mean_all =np.array([0, 0, 0])
        variance_all = np.array([0, 0, 0])
        count=0
        while(count < 1024):
            [single_mean, single_var] = sess.run([image_mean, image_var])
            print('Mean r: ', single_mean[0])
            print('Mean g: ', single_mean[1])
            print('Mean b: ', single_mean[2])
            print('Variance r: ', single_var[0])
            print('Variance g: ', single_var[1])
            print('Variance b: ', single_var[2])
            print('Counter: ',count)
            print(' ')
            mean_all = mean_all+single_mean
            variance_all = variance_all+single_var
            count=count+1
            print('*********************************************************')  
            print('Average Mean: ', mean_all/count)
            print('Average Variance: ', variance_all/count)
