"""
model_train.py

Segmentation network offline trainer.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/scaelles/OSVOS-TensorFlow/blob/master/osvos_parent_demo.py
    Written by Sergi Caelles (scaelles@vision.ee.ethz.ch)
    This file is part of the OSVOS paper presented in:
      Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
      One-Shot Video Object Segmentation
      CVPR 2017
    Unknown code license
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import absolute_import

import os
import sys
import socket
import tensorflow as tf
slim = tf.contrib.slim

# Import model files
# root_folder = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.abspath(root_folder))
import model
import datasets

# User defined parameters
gpu_id = 0

# Training parameters
imagenet_3channels_ckpt = 'models/vgg_16_3chan.ckpt' # downloaded from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
imagenet_4channels_ckpt = 'models/vgg_16_4chan.ckpt'
# logs_path = os.path.join(root_folder, 'models', 'OSVOS_parent_new')
segnet_astream = 'astream'
segnet_fstream = 'fstream'
ckpt_name_astream = segnet_astream + '_parent'
ckpt_name_fstream = segnet_fstream + '_parent'
logs_path_astream = os.path.join('models', ckpt_name_astream)
logs_path_fstream = os.path.join('models', ckpt_name_fstream)
store_memory = True
data_aug = True
iter_mean_grad = 10
max_training_iters_1 = 15000
max_training_iters_2 = 30000
max_training_iters_3 = 50000
save_step = 5000
test_image = None
display_step = 100
ini_learning_rate = 1e-8
boundaries = [10000, 15000, 25000, 30000, 40000]
values = [ini_learning_rate, ini_learning_rate * 0.1, ini_learning_rate, ini_learning_rate * 0.1, ini_learning_rate,
          ini_learning_rate * 0.1]

# Load DAVIS 2016 dataset
options = datasets._DEFAULT_DAVIS16_OPTIONS
# Use non-augmented dataset on laptop
options['data_aug'] = False if socket.gethostname() == 'MSI' else True
dataset_root = 'E:/datasets/davis2016/' if sys.platform.startswith("win") else '/media/EDrive/datasets/davis2016/'
# train_file = dataset_root + 'ImageSets/480p/train_dbg.txt'
train_file = dataset_root + 'ImageSets/480p/train.txt'
dataset = datasets.davis16(train_file, None, dataset_root, options)

# Display dataset configuration
dataset.print_config()

# Train the flow stream branch of the binary segmentation network
imagenet_ckpt = imagenet_3channels_ckpt
segnet_stream = segnet_fstream
ckpt_name= ckpt_name_fstream
logs_path = logs_path_fstream

# Train the network with strong side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 1, learning_rate, logs_path, max_training_iters_1, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, test_image_path=test_image,
                           ckpt_name=ckpt_name)
# Train the network with weak side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(max_training_iters_1, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 2, learning_rate, logs_path, max_training_iters_2, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, resume_training=True,
                           test_image_path=test_image, ckpt_name=ckpt_name)
# Train the network without side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(max_training_iters_2, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 3, learning_rate, logs_path, max_training_iters_3, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, resume_training=True,
                           test_image_path=test_image, ckpt_name=ckpt_name)

# Train the appearance stream branch of the binary segmentation network
imagenet_ckpt = imagenet_4channels_ckpt
segnet_stream = segnet_astream
ckpt_name= ckpt_name_astream
logs_path = logs_path_astream

# Train the network with strong side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 1, learning_rate, logs_path, max_training_iters_1, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, test_image_path=test_image,
                           ckpt_name=ckpt_name)
# Train the network with weak side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(max_training_iters_1, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 2, learning_rate, logs_path, max_training_iters_2, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, resume_training=True,
                           test_image_path=test_image, ckpt_name=ckpt_name)
# Train the network without side outputs supervision
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(max_training_iters_2, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        model.train_parent(dataset, imagenet_ckpt, 3, learning_rate, logs_path, max_training_iters_3, save_step,
                           display_step, global_step, segnet_stream, iter_mean_grad=iter_mean_grad, resume_training=True,
                           test_image_path=test_image, ckpt_name=ckpt_name)

