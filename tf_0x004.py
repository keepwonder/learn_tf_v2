#!usr/bin/env python  
# -*- coding:utf-8 -*-  
# @author: Jone Chiang
# @file  : tf_0x004.py 
# @time  : 2019/07/07 16:01:12

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
(tain_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)





