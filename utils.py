import tensorflow as tf

# H.H Aug 2019

#BN_EPSILON = 0.001
_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 1e-3

def batch_normalization_layer(inputs,training):
    return tf.layers.batch_normalization(
        inputs=inputs, axis=3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)

def nomral_fully(inputs,num_outputs=128):
    features = tf.contrib.layers.fully_connected(
        inputs,
        num_outputs,
        activation_fn=None,#tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None
    )
    return features
def normal_conv(tensor,
                filters=32,
                kernel_size=3,
                strides=1,
                l2_scale=0.0,#5e-5,#0.0, 
                padding='same',
                dilation_rate =(1,1),
                kernel_initializer=tf.contrib.layers.xavier_initializer(), non_lin = tf.nn.relu, bn=False, training=True):

    features = tf.layers.conv2d(tensor, 
                                filters=filters, 
                                kernel_size=kernel_size, 
                                kernel_initializer=kernel_initializer,
                                strides=(strides, strides), 
                                trainable=True, 
                                use_bias=True, 
                                padding=padding,
                                dilation_rate = dilation_rate,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale))
    if bn == False:
        return features
    else:
        bn_layer = batch_normalization_layer(features, training)
        return bn_layer

def normal_deconv(tensor,
                filters=32, trainable = True,
                kernel_size=3,
                strides=1,
                l2_scale=0.0,#1e-4,#5e-5,#0.0, 
                padding='same', 
                kernel_initializer=tf.contrib.layers.xavier_initializer(), bn=False, training=True):

    features = tf.layers.conv2d_transpose(
                                inputs=tensor,
                                filters=filters,
                                kernel_size= kernel_size,
                                strides=(strides, strides),
                                padding=padding,
                                data_format='channels_last',
                                activation=None,
                                use_bias=True,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=tf.zeros_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale),
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                bias_constraint=None,
                                trainable=trainable,
                                name=None,
                                reuse=None)
    if bn == False:
        return features
    else:
        bn_layer = batch_normalization_layer(features, training)
        return bn_layer

def group_norm(features, groups=32,gamma=1,beta=0):
    if gamma > 0: 
        return tf.contrib.layers.group_norm(features,groups=groups, param_initializers={'gamma':tf.constant_initializer(gamma),'beta':tf.constant_initializer(beta)})
    else: 
        return tf.contrib.layers.group_norm(features,groups=groups, param_initializers={'gamma':tf.zeros_initializer(),'beta':tf.zeros_initializer()})

def res_block(x1,filters=64, kernel_size=(3, 3), padding='same', strides = 1, groups=0):
        x = normal_conv(x1 , filters=filters, kernel_size=kernel_size, padding='same', strides = 2)
        if groups>0:
          x = group_norm(x,groups=groups)
        x = tf.nn.relu(x)
        x = normal_conv(x, filters=filters, kernel_size=(1,1), padding='same', strides = 1)
        if groups>0:
          x = group_norm(x,groups=groups)#,gamma=0)
        x_r = normal_conv(x1 , filters=filters, kernel_size=(3,3), padding='same', strides = 2)
        if groups>0:
          x_r = group_norm(x_r,groups=groups)
        return tf.nn.relu(x_r+x)

def res_decode(x1,filters=64, kernel_size=(4, 4), padding='same', strides = 1, groups=0):
        x  = normal_deconv(x1, filters=filters, strides=2, kernel_size=kernel_size, padding='same') 
        if groups>0:
          x = group_norm(x,groups=groups)
        x = tf.nn.leaky_relu(x,alpha=0.2)
        x = normal_conv(x, filters=filters, kernel_size=(1,1), padding='same', strides = 1)
        if groups>0:
          x = group_norm(x,groups=groups)#,gamma=0)
        x_r = normal_deconv(x1 , filters=filters, kernel_size=(4,4), padding='same', strides = 2)
        if groups>0:
          x_r = group_norm(x_r,groups=groups)
        return tf.nn.leaky_relu(x_r+x,alpha=0.2)

def hourglass_decode(features,f=32,groups=0):

    xd  = res_decode(features, filters=f*16, strides=2, kernel_size=(4, 4), padding='same', groups=groups) 
    xd  = res_decode(xd, filters=f*8, strides=2, kernel_size=(4, 4), padding='same', groups=groups) 
    xd  = res_decode(xd, filters=f*4, strides=2, kernel_size=(4, 4), padding='same', groups=groups) 
    xd  = res_decode(xd, filters=f*2, strides=2, kernel_size=(4, 4), padding='same', groups=groups) 
    xd  = res_decode(xd, filters=f, strides=2, kernel_size=(4, 4), padding='same', groups=groups) 

    return xd

def hourglass_encode(features,f=32,groups=0):

    x2_1 = res_block(features , filters=f, kernel_size=(3, 3), padding='same', strides = 2, groups=groups)
    x2_2 = res_block(x2_1 , filters=f*2, kernel_size=(3, 3), padding='same', strides = 2, groups=groups)
    x2_3 = res_block(x2_2 , filters=f*4, kernel_size=(3, 3), padding='same', strides = 2, groups=groups)
    x2_4 = res_block(x2_3 , filters=f*8, kernel_size=(3, 3), padding='same', strides = 2, groups=groups)
    x2_5 = res_block(x2_4 , filters=f*16, kernel_size=(3, 3), padding='same', strides = 2, groups=groups)
    
    return x2_5


def pred_error(features,n_errors=1, filters =16,n_depths=1,groups=0):
    xd = features
    xd_mu0 = normal_conv(xd, filters=filters, kernel_size=(3,3), padding='same')
    if groups>0:
        xd_mu0 = group_norm(xd_mu0, groups=groups)
    xd_mu0 = tf.nn.relu(xd_mu0)
    xd_mu0 = normal_conv(xd_mu0, filters=filters, kernel_size=(3,3), padding='same')
    if groups>0:
        xd_mu0 = group_norm(xd_mu0, groups=groups,gamma=0)
    xd_mu0_ = normal_conv(xd, filters=filters, kernel_size=(5,5), padding='same')
    if groups>0:
        xd_mu0_ = group_norm(xd_mu0_, groups=groups)
    xd_mu0 = tf.nn.relu(xd_mu0_+xd_mu0)
    xd_mu = normal_conv(xd_mu0, filters=n_depths, strides=1, kernel_size=(1, 1), padding='same')
    
    if n_errors>0:
        xd_er0 = normal_conv(xd, filters=filters, kernel_size=(3,3), padding='same')
        if groups>0:
            xd_er0 = group_norm(xd_er0, groups=groups)
        xd_er0 = tf.nn.relu(xd_er0)
        xd_er0 = normal_conv(xd_er0, filters=filters, kernel_size=(3,3), padding='same')
        if groups>0:
            xd_er0 = group_norm(xd_er0, groups=groups,gamma=0)
        xd_er0_ = normal_conv(xd, filters=filters, kernel_size=(5,5), padding='same')
        if groups>0:
            xd_er0_ = group_norm(xd_er0_, groups=groups)
        xd_er0 = tf.nn.relu(xd_er0_+xd_er0)
        xd_er = normal_conv(xd_er0, filters=n_errors, strides=1, kernel_size=(1, 1), padding='same')

        return xd_mu,xd_er
    else:
        return xd_mu,1