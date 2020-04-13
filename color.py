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

class coloring_machine():
    def __init__(self, input_image_size=256, batchsize=5):
        # Hi Mr. Dartfler, the __init__ function is the same as the contructor function in javascript. And self is same as this
        self.batch_size = batchsize
        #batch is the group of training examples in one iteration to update the weights. I want to test the number of training examples in one batch from 4 ~ 10 to see which is better.
