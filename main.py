import argparse
import os
import tensorflow as tf
from cycle_gan import Model
import numpy as np
#from model_for_back_enhanced_multi import conf_net


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='/notebooks/dataset/GANs/spring_winter', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=10000000, help='# of epoch')
parser.add_argument('--dataset_size', dest='ds_size', type=int, default=62, help='# of epoch')
parser.add_argument('--num_gpus', dest='num_gpus', type=int, default=1, help='# of epoch')
parser.add_argument('--max_depth', dest='max_depth', type=float, default=1000.0, help='maximum depth')
parser.add_argument('--mean_depth', dest='mean_depth', type=float, default=74, help='maximum depth')
parser.add_argument('--var_depth', dest='var_depth', type=float, default=1000, help='maximum depth')
parser.add_argument('--max_depth2', dest='max_depth2', type=float, default=10000.0, help='maximum depth')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--decay_step', dest='decay_step', type=int, default=1, help='# of lr decays in each epoch')
parser.add_argument('--decay_rate', dest='decay_rate', type=int, default=0.95, help='lr decay rate')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='/notebooks/project/', help='path of the checkpoint directory')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_test_preds', dest='save_test_preds', type=bool, default=True, help='saves predictions images to the current folder')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True, help='continue training from a checkpoint')
parser.add_argument('--checkpoint_time', dest='checkpoint_time', default='2019_9_17_0_47', help='checkpoint time')
parser.add_argument('--mean_rgb', dest='mean_rgb', type=list, default=[150.43767855, 128.97752175, 144.83988649], help='mean_rgb')
parser.add_argument('--var_rgb', dest='var_rgb', type=list, default=[3886.10923244, 3769.24146542, 4110.46448306], help='var_rgb')


args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    #tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        model.train() if args.phase == 'train' \
            else model.test()

if __name__ == '__main__':
    tf.app.run()