# this is a psuedo code for the coloring project

import tensorflow as tf
import numpy as np
from glob import glob
import math 
import sys
import random 
#import the necessary packages above I think should be useful

'''
To present the nueral network, I want to create an object with methods and attributes. 
The attributes I think could record the values for the filter and similar weights,
and the method will be used to update the weights.
'''

#define a batch normalization object

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

#set a global variable to name each batch normalization object at each convolutional layer
batchnorm_count = 0

def bnreset():
    global batchnorm_count
    batchnorm_count = 0

# I'm learning from others' codes; I don't quite get why people individually divide each batch normalization and define them as objects with different names; Is this the only way to make them trainable? 
def bn(x):
    global batchnorm_count
    batch_object = batch_norm(name=("bn" + str(batchnorm_count)))
    batchnorm_count += 1
    return batch_object(x)

#a modified linear relu function

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

'''
function for (de)convolution layers; note that I may as well just import conv2d from keras which is already defined. But I think the default relu activation is max(0, x) there; I checked 
other coloring nueral network project and they apply relu function as max(x, 0.2x) as above. I don't quite understand but just follow what they did. Will learn about it more later
'''

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv



class coloring_machine():
    def __init__(self, input_image_size=256, batchsize=5):
        # Hi Mr. Dartfler, the __init__ function is the same as the contructor function in javascript. And self is same as this
        self.batch_size = batchsize
        #batch is the group of training examples in one iteration to update the weights. I want to test the number of training examples in one batch from 4 ~ 10 to see which is better.
        self.image_size = input_image_size
        self.output_image_size = input_image_size

        self.line_colordim = 1 
        self.color_colordim = 3
        #here I specify the color dimension for the  line image and the color hints so that I can specify the shape and create the tensorflow values for the image

        self.gf_dim = 64
        self.df_dim = 64
        #img dimension at the convolutonal layers for disciminator and generator

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        #call three batch normalization objects for the discriminator

        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.line_colordim])
        self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.color_colordim])
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.color_colordim])
        #the second part specifies the shape of the tensors | where the line_images/ color_images represent the batch of input training examples 

        combined_preimage = tf.concat([self.line_images, self.color_images], 3)
        #concat the line images and color hints at the color dimension


        self.generated_image = self.generator(combined_preimage)
        #generator() is the generator function yet to be defined

        self.real_coloredimg = tf.concat([combined_preimage, self.real_images], 3)
        self.fake_coloredimg = tf.concat([combined_preimage, self.generated_images], 3)
        #concat at the color dimention to color the preimage for both the generated(fake) and the actual one to feed into the discriminator

        self.disc_true_logits = self.discriminator(self.real_coloredimg, realness=False)
        self.disc_fake_logits = self.discriminator(self.fake_coloredimg, realness=True)
        
        self.disc_loss_real  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_true_logits, tf.ones_like(disc_true_logits)))
        self.disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_logits, tf.zeros_like(disc_fake_logits)))
        '''the discriminator is desgined to return the probabilities of iamge being real for each training examples
        and the function sigmoid_cross_entropy_with_logits maps the difference between generated value and the desired one to 0~1 "input(probabilities, labels)". For the real ones it is all on; and the fake one it is all 0
        reduced mean derives the mean loss among the training examples in this batch
        '''

        self.disc_loss = self.disc_loss_real + self.disc_loss_fake
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_logits, tf.ones_like(disc_fake_logits)))
        #set the loss function for generator and discriminator 

        t_vars = tf.trainable_variables()
        #make a list of trainable variables to divide those in generator and those in discriminator

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        #train the network


        #generator code


    def generator(self, img_in):
        s = self.output_size
        #image size in each convolutional layer
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv') # e1 is (128 x 128 x self.gf_dim)
        e2 = bn(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # e2 is (64 x 64 x self.gf_dim*2)
        e3 = bn(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # e3 is (32 x 32 x self.gf_dim*4)
        e4 = bn(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')) # e4 is (16 x 16 x self.gf_dim*8)
        e5 = bn(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # e5 is (8 x 8 x self.gf_dim*8)


        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e5), [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
        d4 = bn(self.d4)
        d4 = tf.concat(3, [d4, e4])
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
        d5 = bn(self.d5)
        d5 = tf.concat(3, [d5, e3])
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
        d6 = bn(self.d6)
        d6 = tf.concat(3, [d6, e2])
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
        d7 = bn(self.d7)
        d7 = tf.concat(3, [d7, e1])
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_colors], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)

    #for the discriminator
    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv')) # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.df_dim*4)
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))) # h3 is (16 x 16 x self.df_dim*8)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4), h4










