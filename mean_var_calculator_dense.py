import numpy as np
import tensorflow as tf
import pathlib
import random
import os
import matplotlib.pyplot as plt
import argparse
AUTOTUNE = tf.data.experimental.AUTOTUNE
from data_parser import data_parser
from PIL import Image
import matplotlib.pyplot as plt
import cv2

if __name__=='__main__':
    # for verifying the parsed dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #tf.enable_eager_execution()
    with tf.Session() as sess:
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--dataset_dir', dest='dataset_dir', default='/notebooks/dataset/NYU_depth', help='path of the dataset')
        parser.add_argument('--max_depth', dest='max_depth', type=float, default=1, help='maximum depth')
        parser.add_argument('--mean_rgb', dest='mean_rgb', type=list, default=[123.25518942, 106.5671714,  100.99080845], help='mean_rgb')
        parser.add_argument('--var_rgb', dest='var_rgb', type=list, default=[4697.01493893, 4992.08453949, 5351.99687692], help='var_rgb')
        parser.add_argument('--mean_depth', dest='mean_depth', type=float, default=74, help='maximum depth')
        parser.add_argument('--var_depth', dest='var_depth', type=float, default=1000, help='maximum depth')
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

        #mask = tf.where(tf.equal(dense_label, 0), 
        #        tf.zeros_like(dense_label), 
        #        tf.ones_like(dense_label))

        #total_valid_sum = tf.reduce_sum(mask)
        #image_mean = tf.reduce_sum(dense_label)/ total_valid_sum
        
        #dist = tf.square(dense_label - tf.ones_like(dense_label)*image_mean)

        #dist = tf.where(tf.equal(dense_label, 0), 
        #        tf.zeros_like(dist), 
        #        dist)

        #image_var = tf.reduce_sum(dist)/ total_valid_sum

        #sess = tf.Session()
        mean_all =0
        variance_all = 0
        count=0
        depth_all = np.zeros(shape=(480,640))
        while(1):
            [depth] = sess.run([dense_label])
            print (depth.shape)
            depth_all =  np.squeeze(depth)+depth_all

            count=count+1
            depth_avg = depth_all/count
            #img = Image.fromarray(depth_all)
            plt.imsave(arr= depth_avg, fname= './mean_image_vis.jpg' , cmap ='nipy_spectral')
            #plt.imshow(X= depth_avg, cmap ='nipy_spectral')
            to_be_saved = depth_avg.astype(np.uint8)
            cv2.imwrite('./mean_image.jpg',to_be_saved)

            #img.show()
            #img.save('./mean_image.jpg')

            #print('Mean: ', single_mean)
            #print('Variance: ', single_var)
            print('Counter: ',count)
            #print(' ')
            #mean_all = mean_all+single_mean
            #variance_all = variance_all+single_var
            #count=count+1
            #print('*********************************************************')  
            #print('Average Mean: ', mean_all/count)
            #print('Average Variance: ', variance_all/count)
