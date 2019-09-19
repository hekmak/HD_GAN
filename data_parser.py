import numpy as np
import tensorflow as tf
import pathlib
import random
import os
import matplotlib.pyplot as plt
import argparse
import math
AUTOTUNE = tf.data.experimental.AUTOTUNE
s_h = 33*32#64*4#0//4
s_w = 22*32#64*4#0//4
size_h = s_h#1080#//4
size_w = s_w#640#/4
mean_ref_path = './mean_image.jpg'
loc_ref_path = './loc_image.jpg'
class data_parser():
    def __init__(self,args):
        self.train_root_path = args.dataset_dir#os.path.join(args.dataset_dir,'')
        self.args = args

    def create_dataset(self,shuffle=True):
        train_root = pathlib.Path(self.train_root_path)

        # image 
        rgb_paths_train = list(train_root.glob('*.jpg'))

        # convert to string
        rgb_paths_train = sorted([str(path) for path in rgb_paths_train])

        print (len(rgb_paths_train))

        # shuffling sparse and dense depths with a same order
        random.shuffle(rgb_paths_train)
        #rgb_paths_train = list(rgb_paths_train)

        # train
        rgb_ds_train = tf.data.Dataset.from_tensor_slices(rgb_paths_train)

        ds_train = rgb_ds_train.map(self.load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

        return ds_train

    def preprocess_image(self,image):
        img_tensor = tf.image.decode_jpeg(image)
        img_tensor = tf.cast(img_tensor,dtype=tf.dtypes.float32)


        #print(tf.shape(img_tensor))
        #quit()
        #img_tensor = tf.image.resize_images(img_tensor,(size_h+10, size_w+100000),preserve_aspect_ratio=True)
        #image = tf.image.random_crop(img_tensor,(size_h, size_w,3))
        image = tf.image.resize_images(
            img_tensor,(s_h,s_w))
        image = tf.reshape(image,[1,s_h, s_w,3]) 

        #image = tf.image.random_flip_left_right(image)
        r,g,b = tf.split(image,num_or_size_splits=3,axis=3)
        image = tf.concat([r,g,b],axis=3)

        image=tf.reshape(image,[s_h, s_w, 3]) 

        # normalize zero mean variance 1
        
        #r,g,b= tf.split(image,num_or_size_splits=3,axis=2)
        #r = (r- self.args.mean_rgb[0])/math.sqrt(self.args.var_rgb[0])
        #g = (g- self.args.mean_rgb[1])/math.sqrt(self.args.var_rgb[1])
        #b = (b- self.args.mean_rgb[2])/math.sqrt(self.args.var_rgb[2])
        #image = tf.concat([r,g,b],axis=2)
        image = image/128.0 - 1
        
        return image

    def load_and_preprocess_image(self,path_img):
        image = tf.read_file(path_img)
        
        return self.preprocess_image(image)


if __name__=='__main__':
    # for verifying the parsed dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='/notebooks/dataset/GANs/spring_winter', help='path of the dataset')
    parser.add_argument('--mean_rgb', dest='mean_rgb', type=list, default=[123.67608894, 106.15077857, 101.53614825], help='maximum depth')
    parser.add_argument('--var_rgb', dest='var_rgb', type=list, default=[4667.6736461,  4984.61869816, 5358.27851633], help='maximum depth')
    parser.add_argument('--mean_depth', dest='mean_depth', type=float, default=2842, help='maximum depth')
    parser.add_argument('--var_depth', dest='var_depth', type=float, default=1275316, help='maximum depth')
    args = parser.parse_args()
    ds = data_parser(args=args)
    ds_train = ds.create_dataset()
    plt.figure(figsize=(8,8))
    n=0
    for image in ds_train.take(4):
        plt.subplot(4,3,n+1)
        plt.imshow(np.squeeze(image))
        plt.subplot(4,3,n+2)
        plt.imshow(np.squeeze(image))
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        n=n+2
    plt.show()