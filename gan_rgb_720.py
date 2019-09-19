import tensorflow as tf
from data_parser import data_parser
from utils import *
import math
import datetime
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.stats as stats



AUTOTUNE = tf.data.experimental.AUTOTUNE


class Model():
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.shuffle_buffer_size = 100 # fix later
        self.previous_RMSE = 1e10
        self.current_lr = args.lr
        

    def read_data(self):
        dp = data_parser(args=self.args)
        self.ds_train = dp.create_dataset()

    def generator(self, tf_input ,reuse=False ):
        with tf.variable_scope('ham_generator', reuse=reuse):
            projected = normal_conv(tf_input, filters=1024*33*22, kernel_size=(10,10), padding='same', strides = 10)
            projected = tf.reshape(projected,[self.args.batch_size,33,22,1024])
            x2 = hourglass_decode(projected, f=64,groups=32)
            print('**************\n',x2)
            pred, _ = pred_error(x2,n_errors=0, filters =48,n_depths=3,groups=16)
            pred = tf.tanh(pred)
            return tf.identity(pred,name='pred_gan')

    def discriminator(self, tf_input ,reuse=False ):
        with tf.variable_scope('ham_discriminator', reuse=reuse):
            x2 = hourglass_encode(tf_input, f=64,groups=32)
            pred, _ = pred_error(x2,n_errors=0, filters =16,n_depths=1,groups=16)
            pred = tf.layers.flatten(pred)

            pred = nomral_fully(pred,1)
            return tf.identity(pred,name='pred_gan')

    def squared_loss(self,x, y, mask):
        return tf.reduce_mean(tf.losses.mean_squared_error(predictions=x,
                                                             labels=y,
                                                             weights=mask))

    def compute_loss(self,disc_logits_real,disc_logits_fake):

        #loss_discriminator_real = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_logits_real)*0.9,logits=disc_logits_real))
        #loss_discriminator_fake = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_logits_real),logits=disc_logits_fake))
        loss_discriminator_real = tf.losses.hinge_loss(labels=tf.ones_like(disc_logits_real),logits=disc_logits_real)
        loss_discriminator_fake = tf.losses.hinge_loss(labels=tf.zeros_like(disc_logits_real),logits=disc_logits_fake)
        loss_discriminator = loss_discriminator_real+loss_discriminator_fake
        loss_generator = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_logits_real),logits=disc_logits_fake))
        # normalizing losses by their value  
        ''''
        Aloss_reduce1 =  tf.Variable(initial_value=0.0,trainable=False)
        new_loss1 =  tf.Variable(initial_value=0.0,trainable=False)
        Aloss_reduce2 =  tf.Variable(initial_value=0.0,trainable=False)
        new_loss2 =  tf.Variable(initial_value=0.0,trainable=False)
        Aloss_reduce3 =  tf.Variable(initial_value=0.0,trainable=False)
        new_loss3 =  tf.Variable(initial_value=0.0,trainable=False)

        new_loss1 = new_loss1.assign(tf.to_float(loss_discriminator))
        Aloss_reduce1 =  Aloss_reduce1.assign( tf.to_float( tf.abs(new_loss1) ))
        new_loss2 = new_loss2.assign(tf.to_float(loss_generator))
        Aloss_reduce2 =  Aloss_reduce2.assign( tf.to_float( tf.abs(new_loss2) ))

        Aloss_reduce2 = tf.where(tf.equal(Aloss_reduce2, 0), 
                        tf.ones_like(Aloss_reduce2), 
                        Aloss_reduce2)
        Aloss_reduce1 = tf.where(tf.equal(Aloss_reduce1, 0), 
                        tf.ones_like(Aloss_reduce1), 
                        Aloss_reduce1)
        '''
        loss_discriminator_ = loss_discriminator#/Aloss_reduce1
        loss_generator_ = loss_generator#/Aloss_reduce2

        print ('loss_discriminator****************',loss_discriminator,loss_generator)
        return loss_discriminator_, loss_generator_, loss_discriminator, loss_generator

    def input_fn(self,train=True):
        if train:
            ds = self.ds_train.shuffle(buffer_size=self.shuffle_buffer_size)
            ds = ds.repeat()
            ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(self.args.batch_size))
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            # creating the training iterator
            iter = ds.make_one_shot_iterator()
        else:
            # creating the validation iterator
            ds_val = self.ds_val.repeat()
            ds_val = ds_val.apply(tf.contrib.data.batch_and_drop_remainder(self.args.batch_size))
            iter = ds_val.make_one_shot_iterator()
        return iter

    def tower_loss(self, scope, real_image, gpu_name):

        # build model and loss (train)
        #noise = tf.random.normal([self.args.batch_size,100])
        #noise = tf.random.uniform([self.args.batch_size,10,10,1],minval=-1,maxval=1)
        noise = tf.random.truncated_normal([self.args.batch_size,10,10,1])

        fake_image = self.generator(noise)
        disc_logits_real = self.discriminator(real_image)
        disc_logits_fake = self.discriminator(fake_image,reuse=True)

        disc_logits = tf.concat(axis=0, values=[disc_logits_real,disc_logits_fake])

        disc_logits_real,disc_logits_fake = tf.split(disc_logits,num_or_size_splits=2,axis=0)
        loss_discriminator, loss_generator, loss_discriminator_, loss_generator_ = self.compute_loss(disc_logits_real,disc_logits_fake)


        # create a summary writer  
        max_outputs = 3      
        tf.summary.image(gpu_name+"/Input/real_image/", real_image,max_outputs=max_outputs)
        tf.summary.image(gpu_name+"/Prediction/fake_image/", fake_image,max_outputs=max_outputs)
        tf.summary.image(gpu_name+"/Input/noise/", noise,max_outputs=max_outputs)
        tf.summary.scalar(gpu_name+"/loss_discriminator_/", loss_discriminator_)
        tf.summary.scalar(gpu_name+"/loss_generator_/", loss_generator_)        
        tf.summary.scalar(gpu_name+"/loss_discriminator/", loss_discriminator)
        tf.summary.scalar(gpu_name+"/loss_generator/", loss_generator)

        return loss_discriminator, loss_generator,fake_image

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def multi_gpu(self,iter,global_step,learning_rate):
        tower_grads_disc = []
        tower_grads_gen = []
        RMSEs =  []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.args.num_gpus):
                real_image = iter.get_next()
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('ConfNet', i)) as scope:
                        # all towers.
                        gpu_name = 'GPU'+str(i)
                        loss_discriminator, loss_generator,fake_image = self.tower_loss(scope, real_image,gpu_name)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        # Calculate the gradients for the batch of data on this tower.
                        disc_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ham_discriminator')
                        gen_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ham_generator')
                        grads_disc = self.optimizer_disc.compute_gradients(loss_discriminator,disc_vars)
                        grads_gen = self.optimizer_gen.compute_gradients(loss_generator,gen_vars)
                        # Keep track of the gradients across all towers.
                        tower_grads_disc.append(grads_disc)    
                        tower_grads_gen.append(grads_gen)    
                        tf.summary.scalar(gpu_name+"/global_step/", global_step)
                        tf.summary.scalar(gpu_name+"/lr/", learning_rate)                       

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads_disc = self.average_gradients(tower_grads_disc)
        grads_gen = self.average_gradients(tower_grads_gen)

        # Apply the gradients to adjust the shared variables.
        train_op_disc = self.optimizer_disc.apply_gradients(grads_disc, global_step=global_step)
        train_op_gen = self.optimizer_gen.apply_gradients(grads_gen, global_step=global_step)


        return train_op_disc,train_op_gen,fake_image

    def train(self):
        global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

        self.read_data()
        iter = self.input_fn(train=True)

        # learning rate and optimizer
        '''
        decay_steps = int(self.args.ds_size/(self.args.batch_size*self.args.num_gpus)/self.args.decay_step)
        lr = tf.train.exponential_decay(self.args.lr,
                                        global_step,
                                        decay_steps,
                                        self.args.decay_rate,
                                        staircase=True)
        '''
        learning_rate = tf.placeholder(tf.float32, shape=[])

        #self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
        self.optimizer_gen = tf.train.AdamOptimizer(
            learning_rate=learning_rate,beta1=0.5)        
        self.optimizer_disc = tf.train.AdamOptimizer(
            learning_rate=3*learning_rate,beta1=0.5)

        train_op_disc,train_op_gen,fake_image = self.multi_gpu(iter,global_step,learning_rate)
        # saver
        saver = tf.train.Saver(tf.global_variables())
        now = datetime.datetime.now()
        timak =str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)
        log_path = './logs/'+timak
        if self.args.continue_train:
            checkpoints_path = os.path.join('./checkpoints/' + self.args.checkpoint_time) 
            try:
                saver.restore(self.sess, tf.train.latest_checkpoint(checkpoints_path))
                print("Restored the checkpoint!")
            except:
                print("Can't load the checkpoint!")
        else:
            checkpoints_path = './checkpoints/' + timak 
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(log_path, self.sess.graph)
        merged = tf.summary.merge_all()

        # training loop
        epoch_num=0 
        real_batch_size = self.args.batch_size*self.args.num_gpus
        for e in range(self.args.epoch):
            for i in range(self.args.ds_size//real_batch_size): 
                # linear warm-up
                #if e==0:
                #    if i < 100:#200*real_batch_size:
                #        self.current_lr = self.args.lr/20
                #    elif i < 200:#200*real_batch_size:
                #        self.current_lr = self.args.lr/4
                #    elif i < 300:#200*real_batch_size:
                #        self.current_lr = self.args.lr/2
                #    elif i < 400:#200*real_batch_size:
                #        self.current_lr = self.args.lr/1.3
                #    else:#200*real_batch_size:
                #        self.current_lr = self.args.lr

                self.current_lr = self.args.lr
                [_,__, summary,fake_image_to_save] = self.sess.run([train_op_disc,train_op_gen, merged,fake_image], feed_dict={learning_rate: self.current_lr})
                to_be_saved_min = np.squeeze( ((fake_image_to_save+1)*128)/256.0 )
                step = (epoch_num*(self.args.ds_size//real_batch_size) + i)
                if i%50 == 0:
                    self.writer.add_summary(summary, step)
                
                if i%(self.args.ds_size*20) == 0:
                    path = os.path.join(checkpoints_path, 'ham_gan')
                    saver.save(self.sess, path,global_step= step)
                    #self.writer.add_summary(RMSE_all, i)

            epoch_num +=1

    def test(self):
        out_pathes = ['/notebooks/project/predictions/input',
                    '/notebooks/project/predictions/miniature'] 
        for path in out_pathes:
          if not os.path.exists(path):
            os.makedirs(path)

        # creating the input noise
        #noise = tf.random.truncated_normal([self.args.batch_size,10,10,1])
        noise= tf.placeholder(tf.float32, shape=[self.args.batch_size,10,10,1])
        fake_image = self.generator(noise)

        # saver
        saver = tf.train.Saver()
        checkpoints_path = os.path.join('./checkpoints/' + self.args.checkpoint_time) 
        try:
            saver.restore(self.sess, tf.train.latest_checkpoint(checkpoints_path))
        except:
            print("Can't load the checkpoint")
        RMSE = 0
        mean=0.0
        stddev=1.0
        for j in range( int(100) ):
            # trunc normal in scipy
            noisak = stats.truncnorm.rvs(mean, stddev, size=[self.args.batch_size,10,10,1])
            [in_noise,pred_min] = self.sess.run([noise,fake_image],feed_dict={noise:noisak})

            for i in range(self.args.batch_size):
                image_num = "{:05d}.png".format(j*self.args.batch_size+i)
                if self.args.save_test_preds:
                    fname_min= os.path.join('/notebooks/project/predictions/miniature',image_num)
                    fname_in= os.path.join('/notebooks/project/predictions/input',image_num)
                    to_be_saved_min = np.squeeze(pred_min)
                    to_be_saved_in = np.squeeze(in_noise)
                    plt.imsave(arr= np.squeeze(to_be_saved_min), fname= fname_min )#, cmap ='nipy_spectral')
                    plt.imsave(arr= np.squeeze(to_be_saved_in), fname= fname_in , cmap ='gray')
                print('Processed image ' + image_num + '!')