"""
model_finetune.py

Segmentation network online trainer.

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
import model
import datasets

# Model paths
seq_name = "car-shadow"
segnet_stream = 'astream'
parent_path = 'models/' + segnet_stream + '_parent/' + segnet_stream + '_parent-50000'
ckpt_name = segnet_stream + '_' + seq_name
logs_path = 'models/' + ckpt_name

# Online training parameters
gpu_id = 0
max_training_iters = 500
learning_rate = 1e-8
save_step = max_training_iters
side_supervision = 3
display_step = 10

# Load the DAVIS 2016 sequence
options = datasets._DEFAULT_DAVIS16_OPTIONS
options['use_cache'] = False
options['data_aug'] = True
# Set the following to wherever you have downloaded the DAVIS 2016 dataset
dataset_root = 'E:/datasets/davis2016/' if sys.platform.startswith("win") else '/media/EDrive/datasets/davis2016/'
test_frames = sorted(os.listdir(dataset_root + 'JPEGImages/480p/' + seq_name))
test_imgs = [dataset_root + 'JPEGImages/480p/' + seq_name + frame for frame in test_frames]
train_imgs = [dataset_root + 'JPEGImages/480p/' + seq_name + '00000.jpg',
              dataset_root + 'Annotations/480p' + seq_name + '00000.png']
dataset = datasets.davis16(train_imgs, test_imgs, dataset_root, options)

# Display dataset configuration
dataset.print_config()

# Finetune the appearance stream of the binary segmentation network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        model.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                             save_step, display_step, global_step, segnet_stream, iter_mean_grad=1, ckpt_name=ckpt_name)

# Model paths
segnet_stream = 'fstream'
parent_path = 'models/' + segnet_stream + '_parent/' + segnet_stream + '_parent-50000'
ckpt_name = segnet_stream + '_' + seq_name
logs_path = 'models/' + ckpt_name

# Finetune the flow stream branch of the binary segmentation network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        model.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                             save_step, display_step, global_step, segnet_stream, iter_mean_grad=1, ckpt_name=ckpt_name)

