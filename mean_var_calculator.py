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
        parser.add_argument('--dataset_dir', dest='dataset_dir', default='/notebooks/dataset/NYU_depth', help='path of the dataset')
        parser.add_argument('--max_depth', dest='max_depth', type=float, default=1, help='maximum depth')
        args = parser.parse_args()
        ds = data_parser(args=args)
        ds_train, ds_val = ds.create_dataset()

        ds = ds_train.shuffle(buffer_size=1000)
        #ds = ds.batch(self.args.batch_size)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(1))
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        # creating the training iterator
        iter = ds.make_one_shot_iterator()

        sparse_input, dense_label = iter.get_next()

        mask = tf.where(tf.equal(dense_label, 0), 
                tf.zeros_like(dense_label), 
                tf.ones_like(dense_label))

        total_valid_sum = tf.reduce_sum(mask)
        image_mean = tf.reduce_sum(dense_label)/ total_valid_sum
        
        dist = tf.square(dense_label - tf.ones_like(dense_label)*image_mean)

        dist = tf.where(tf.equal(dense_label, 0), 
                tf.zeros_like(dist), 
                dist)

        image_var = tf.reduce_sum(dist)/ total_valid_sum

        #sess = tf.Session()
        mean_all =0
        variance_all = 0
        count=0
        while(1):
            [single_mean, single_var] = sess.run([image_mean, image_var])
            print('Mean: ', single_mean)
            print('Variance: ', single_var)
            print('Counter: ',count)
            print(' ')
            mean_all = mean_all+single_mean
            variance_all = variance_all+single_var
            count=count+1
            print('*********************************************************')  
            print('Average Mean: ', mean_all/count)
            print('Average Variance: ', variance_all/count)
