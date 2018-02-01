"""
datasets.py

Davis 2016 dataset utility functions and classes.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/scaelles/OSVOS-TensorFlow/blob/master/dataset.py
    Written by Sergi Caelles (scaelles@vision.ee.ethz.ch)
    This file is part of the OSVOS paper presented in:
      Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
      One-Shot Video Object Segmentation
      CVPR 2017
    Unknown code license
    
References for future work:
  - https://github.com/CharlesShang/FastMaskRCNN/blob/master/libs/datasets/coco.py
    Copyright (c) 2017 Charles Shang / Written by Charles Shang
    Licensed under the Apache License, Version 2.0, January 2004
  - https://github.com/matterport/Mask_RCNN/blob/master/utils.py
    Copyright (c) 2017 Matterport, Inc. / Written by Waleed Abdulla
    Licensed under the MIT License
  - https://github.com/ferreirafabio/video2tfrecords/blob/master/video2tfrecords.py
    Copyright (c) 2017 Fábio Ferreira / Written Fábio Ferreira
    Licensed under the MIT License
  - https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/tf_records.py
    https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/recipes/pascal_voc/convert_pascal_voc_to_tfrecords.ipynb
    Copyright (c) 2017 Daniil Pakhomov / Written by Daniil Pakhomov
    Licensed under the MIT License
  - http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    How to write into and read from a tfrecords file in TensorFlow
    Writeen by Hadi Kazemi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import os
import numpy as np
import sys
import random
from tqdm import tqdm, trange
# from tqdm import tqdm_notebook as tqdm, trange
# from tqdm import tqdm_notebook, trange
from IPython.display import HTML
import io, base64
import imageio
import bboxes
import optflow
import visualize

_DEFAULT_DAVIS16_OPTIONS = {
    'in_memory': True,
    'data_aug': True,
    'use_cache': True,
    'use_optical_flow': True,
    'use_warped_masks': True,
    'use_bboxes': True,
    'optical_flow_mgr': 'pyflow'
}

class davis16(object):
    def __init__(self, train_list, test_list, dataset_root, options=_DEFAULT_DAVIS16_OPTIONS):
        """Initialize the Dataset object
        Args:
        train_list: TXT file or list with the paths of the images to use for training (Images must be between 0 and 255)
        test_list: TXT file or list with the paths of the images to use for testing (Images must be between 0 and 255)
        dataset_root: Path to the root of the Database
        Options:
          in_memory: True stores all the training images, False loads at runtime the images
          data_aug: True adds augmented data to training set
          use_cache: True stores training files and augmented versions in npy file
          use_warped_masks: True generates warped predictions of the masks, using optical flow
          use_bboxes: True computes bounding boxes for masks
          use_flow_magnitues: True computes magnitudes of forward and backward flows
        Returns:
        """
        # Check settings
        if not options['in_memory'] and options['data_aug']:
            sys.stderr.write('Online data augmentation not supported when the data is not stored in memory!')
            sys.exit()

        # Initialize data members
        self.dataset_root = dataset_root
        self.options = options
        self.images_train = []
        self.images_train_path = []
        self.masks_train = []
        self.masks_train_path = []
        self.cache_needs_refresh = False
        self.videos = {}
        self.video_names = []
        self.video_frame_idx = {}
        self.num_videos = 0
        self.flow_norms_train = [] if self.options['use_optical_flow'] else None
        self.warped_prev_masks_train = [] if self.options['use_warped_masks'] else None
        self.masks_bboxes_train = [] if self.options['use_bboxes'] else None

        # Define types of data augmentation
        self.data_aug_scales = [0.5, 0.8]
        self.data_aug_flip = True

        # Load training images and masks (or their paths)
        print('Initializing dataset...')
        self._load_training_data(train_list)

        # Generate motion flows for each frame, warp masks, compute their bboxes
        if self.options['use_optical_flow'] and len(self.flow_norms_train) == 0:
            if self.options['optical_flow_mgr'] == 'pyflow':
                self.optflow = optflow.optflow()
            self._use_optical_flows()
        if self.options['use_bboxes'] and len(self.masks_bboxes_train) == 0:
            self._use_mask_bboxes()

        # Write ndarrays to cache, if requested
        if self.options['use_cache'] and self.cache_needs_refresh:
            self._write_to_cache()

        # Load testing images (or paths)
        self._load_testing_data(test_list)

        print('...done initializing Dataset')

        # Init parameters
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = max(len(self.images_train_path), len(self.images_train))
        self.test_size = len(self.images_test_path)
        self.train_idx = np.arange(self.train_size)
        np.random.seed(1969)
        np.random.shuffle(self.train_idx)

        # Test debugging code
        # self._test_debug()
        # self._fix_broken_flows()

    ###
    ### Cache Management (using npy files)
    ###

    def _build_cache_names(self):
        """Construct the list of file names for the cache
         """
        cache_basename = os.path.basename(os.path.normpath(self.dataset_root))
        self.videos_train_cache = os.path.join(self.dataset_root, cache_basename + '_videos_train.npy')
        self.video_frame_idx_train_cache = os.path.join(self.dataset_root, cache_basename + '_video_frame_idx_train.npy')
        self.images_train_cache = os.path.join(self.dataset_root, cache_basename + '_images_train.npy')
        self.images_train_path_cache = os.path.join(self.dataset_root, cache_basename + '_images_train_path.npy')
        self.masks_train_cache = os.path.join(self.dataset_root, cache_basename + '_masks_train.npy')
        self.masks_train_path_cache = os.path.join(self.dataset_root, cache_basename + '_masks_train_path.npy')
        if self.options['use_optical_flow']:
            self.flow_norms_train_cache = os.path.join(self.dataset_root, cache_basename + '_flow_norms_train.npy')
        else:
            self.flow_norms_train_cache = None
        if self.options['use_warped_masks']:
            self.warped_prev_masks_train_cache = os.path.join(self.dataset_root, cache_basename + '_warped_prev_masks_train.npy')
        else:
            self.warped_prev_masks_train_cache = None
        if self.options['use_bboxes']:
            self.masks_bboxes_train_cache = os.path.join(self.dataset_root, cache_basename + '_masks_bboxes_train.npy')
        else:
            self.masks_bboxes_train_cache = None

    def _print_cache_names(self):
        """Print the names of the files used by the cache
         """
        print("Cache files:")
        if self.videos_train_cache:
            print("   videos container: %s" % self.videos_train_cache)
        if self.video_frame_idx_train_cache:
            print("   video_frame_idx container: %s" % self.video_frame_idx_train_cache)
        if self.images_train_cache:
            print("   images_train container: %s" % self.images_train_cache)
        if self.images_train_path_cache:
            print("   images_train_path container: %s" % self.images_train_path_cache)
        if self.masks_train_cache:
            print("   masks_train container: %s" % self.masks_train_cache)
        if self.masks_train_path_cache:
            print("   masks_train_path container: %s" % self.masks_train_path_cache)
        if self.flow_norms_train_cache:
            print("   flow_norms_train container: %s" % self.flow_norms_train_cache)
        if self.warped_prev_masks_train_cache:
            print("   warped_prev_masks_train container: %s" % self.warped_prev_masks_train_cache)
        if self.masks_bboxes_train_cache:
            print("   masks_bboxes_train container: %s" % self.masks_bboxes_train_cache)

    def _write_to_cache(self):
        """Write (augmented) images and masks to disk
         """
        print("Writing ndarrays to cache...")

        if self.videos_train_cache:
            np.save(self.videos_train_cache, self.videos)
        if self.video_frame_idx_train_cache:
            np.save(self.video_frame_idx_train_cache, self.video_frame_idx)
        if self.images_train_cache:
            np.save(self.images_train_cache, self.images_train)
        if self.images_train_path_cache:
            np.save(self.images_train_path_cache, self.images_train_path)
        if self.masks_train_cache:
            np.save(self.masks_train_cache, self.masks_train)
        if self.masks_train_path_cache:
            np.save(self.masks_train_path_cache, self.masks_train_path)
        if self.flow_norms_train:
            np.save(self.flow_norms_train_cache, self.flow_norms_train)
        if self.warped_prev_masks_train:
            np.save(self.warped_prev_masks_train_cache, self.warped_prev_masks_train)
        if self.masks_bboxes_train:
            np.save(self.masks_bboxes_train_cache, self.masks_bboxes_train)

        print("...done writing to cache.")

    def _load_from_cache(self):
        """Load (augmented) images and masks from disk
        Returns True if loaded from cache, False otherwise
        """
        self._build_cache_names()
        self._print_cache_names()

        # These four should be handled together
        videos_exist = os.path.exists(self.videos_train_cache) and os.path.exists(self.video_frame_idx_train_cache)
        images_exist = os.path.exists(self.images_train_cache)
        images_path_exist = os.path.exists(self.images_train_path_cache)
        masks_exist = os.path.exists(self.masks_train_cache)
        masks_path_exist = os.path.exists(self.masks_train_path_cache)
        if not videos_exist or not images_exist or not images_path_exist or not masks_exist or not masks_path_exist:
            print("   a cache file does not exist")
            self.cache_needs_refresh = True
            return False

        # Only cache flow norms, warped previous masks, and bounding boxes
        # These are the only buffers that MaskRNN sees as input to its CNNs
        if self.options['use_optical_flow']:
            if not os.path.exists(self.flow_norms_train_cache):
                print("   a cache file does not exist")
                self.cache_needs_refresh = True
                return False
        if self.options['use_warped_masks'] and not os.path.exists(self.warped_prev_masks_train_cache):
            print("   a cache file does not exist")
            self.cache_needs_refresh = True
            return False
        if self.options['use_bboxes'] and not os.path.exists(self.masks_bboxes_train_cache):
            print("   a cache file does not exist")
            self.cache_needs_refresh = True
            return False

        print("Loading ndarrays from cache...")

        print(' videos container...', end="")
        self.videos = np.load(self.videos_train_cache).item()
        print(' done.')
        self.num_videos = len(self.videos)
        print(' video_frame_idx container...', end="")
        self.video_frame_idx = np.load(self.video_frame_idx_train_cache).item()
        print(' done.')
        self.num_frames_train = len(self.video_frame_idx)
        print(' images_train container...', end="")
        self.images_train = np.load(self.images_train_cache)
        print(' done.')
        assert(self.num_frames_train == len(self.images_train))
        print(' images_train_path container...', end="")
        self.images_train_path = np.load(self.images_train_path_cache)
        print(' done.')
        print(' masks_train container...', end="")
        self.masks_train = np.load(self.masks_train_cache)
        print(' done.')
        assert(self.num_frames_train == len(self.masks_train))
        print(' masks_train_path container...', end="")
        self.masks_train_path = np.load(self.masks_train_path_cache)
        print(' done.')
        if self.options['use_optical_flow']:
            print(' flow_norms_train container...', end="")
            self.flow_norms_train = np.load(self.flow_norms_train_cache)
            print(' done.')
            assert (self.num_frames_train == len(self.flow_norms_train))
        if self.options['use_warped_masks']:
            print(' warped_prev_masks_train container...', end="")
            self.warped_prev_masks_train = np.load(self.warped_prev_masks_train_cache)
            print(' done.')
            assert (self.num_frames_train == len(self.warped_prev_masks_train))
        if self.options['use_bboxes']:
            print(' masks_bboxes_train container...', end="")
            self.masks_bboxes_train = np.load(self.masks_bboxes_train_cache)
            print(' done.')
            assert (self.num_frames_train == len(self.masks_bboxes_train))

        print("...done loading from cache.")
        return True

    ###
    ### TODO TFRecords helpers
    ### See:
    ### https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/base.py
    ### https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/loader.py
    ### https://github.com/kwotsin/create_tfrecords
    ### https://kwotsin.github.io/tech/2017/01/29/tfrecords.html
    ### http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/
    ### E:\repos\models-master\research\inception\inception\data\build_imagenet_data.py
    ### E:\repos\models-master\research\object_detection\dataset_tools\create_kitti_tf_record.py
    ###
    def _load_from_tfrecords(self):
        # TODO _load_from_tfrecords
        pass

    def _write_to_tfrecords(self):
        # TODO _write_to_tfrecords
        pass

    ###
    ### Load Training and Testing Date (+Augment)
    ###
    def _load_training_data(self, train_list):
        """Load training images and masks (or path to them) from disk
        Also perform data augmentation if reguested
        """
        print(train_list)
        if not isinstance(train_list, list) and train_list:
            with open(train_list) as t:
                train_paths = t.readlines()
        elif isinstance(train_list, list):
            train_paths = train_list
        else:
            train_paths = []

        if not self.options['use_cache'] or not self._load_from_cache():
            self.cache_needs_refresh = True
            if self.options['in_memory']:
                base_video_names = []

                # Get the list of videos and their images
                for line in train_paths:
                    # img_file = os.path.join(self.dataset_root, str(line.split()[0]))
                    # mask_file = os.path.join(self.dataset_root, str(line.split()[1]))
                    img_file = self.dataset_root + str(line.split()[0])
                    mask_file = self.dataset_root + str(line.split()[1])
                    base_video_name = os.path.basename(os.path.dirname(img_file))
                    video_name = base_video_name
                    if video_name in self.videos:
                        self.videos[video_name] += 1
                        frame_idx += 1
                    else:
                        self.num_videos += 1
                        self.videos[video_name] = 1
                        frame_idx = 0
                    video_frame = video_name + '_' + str(frame_idx)
                    self.video_frame_idx[video_frame] = len(self.images_train_path)
                    base_video_names.append(base_video_name)
                    self.images_train_path.append(img_file)
                    self.masks_train_path.append(mask_file)

                print('Loading training images and masks...')
                self.num_original_frames = len(self.images_train_path)
                for frame in trange(self.num_original_frames, ncols=80):
                    img = Image.open(self.images_train_path[frame])
                    mask = Image.open(self.masks_train_path[frame])
                    img.load()
                    mask.load()
                    # Images such as E:\datasets\davis2016\Annotations\480p\bear\00077.png have
                    # a different format -> convert from that format (i.e., "LA") to "L"
                    if mask.mode == "LA":
                        mask = mask.convert("L")
                        tqdm.write('warning: converted {} from format "LA" to {}'.format(self.masks_train_path[frame], mask.mode))
                        # if 76 <= frame <= 78:
                        # print("Mask {} of mode {} is {}".format(frame, mask.mode, ))
                    self.images_train.append(img)
                    self.masks_train.append(mask)
                print('...done loading training images and masks.')

                # Perform scaling data augmentation on frames/masks
                if self.options['data_aug']:
                    print('Performing scaling data augmentation on frames/masks...')
                    for scale_idx in trange(len(self.data_aug_scales), ncols=80):
                        scale = self.data_aug_scales[scale_idx]
                        for frame in trange(self.num_original_frames, ncols=80):
                            base_video_name = base_video_names[frame]
                            video_name = base_video_name + '_sc_' + str(scale)
                            if video_name in self.videos:
                                self.videos[video_name] += 1
                                frame_idx += 1
                            else:
                                self.num_videos += 1
                                self.videos[video_name] = 1
                                frame_idx = 0
                            img = self.images_train[frame]
                            img_size = tuple([int(img.size[0] * scale), int(img.size[1] * scale)])
                            img_sc = img.resize(img_size)
                            mask = self.masks_train[frame]
                            mask_sc = mask.resize(img_size)
                            video_frame = video_name + '_' + str(frame_idx)
                            self.video_frame_idx[video_frame] = len(self.images_train)
                            base_video_names.append(video_name)
                            self.images_train.append(img_sc)
                            self.masks_train.append(mask_sc)
                    print('... done performing scaling data augmentation on frames/masks.')

                    if self.data_aug_flip:
                        print('Performing flipping data augmentation on frames/masks...')
                        for frame in trange(len(self.images_train), ncols=80):
                            base_video_name =base_video_names[frame]
                            video_name = base_video_name + '_fl'
                            if video_name in self.videos:
                                self.videos[video_name] += 1
                                frame_idx += 1
                            else:
                                self.num_videos += 1
                                self.videos[video_name] = 1
                                frame_idx = 0
                            img = self.images_train[frame]
                            img_fl = img.transpose(Image.FLIP_LEFT_RIGHT)
                            mask = self.masks_train[frame]
                            mask_fl = mask.transpose(Image.FLIP_LEFT_RIGHT)
                            video_frame = video_name + '_' + str(frame_idx)
                            self.video_frame_idx[video_frame] = len(self.images_train)
                            base_video_names.append(video_name)
                            self.images_train.append(img_fl)
                            self.masks_train.append(mask_fl)
                        print('... done performing flipping data augmentation on frames/masks.')

                # Convert all images and masks to numpy arrays
                print('Converting images and masks to numpy arrays...')
                for frame in trange(len(self.images_train), ncols=80):
                    self.images_train[frame] = np.array(self.images_train[frame], dtype=np.uint8)
                    assert(len(self.images_train[frame].shape) == 3)
                    mask = np.array(self.masks_train[frame], dtype=np.uint8)
                    mask = np.expand_dims(mask, axis=-1)
                    self.masks_train[frame] = mask
                    # print(self.masks_train[frame].shape)
                    assert(len(self.masks_train[frame].shape) == 3 and mask.shape[2]==1)
                print('...done converting images and masks to numpy arrays.')
            else:
               for line in train_paths:
                    # img_file = os.path.join(self.dataset_root, str(line.split()[0]))
                    # mask_file = os.path.join(self.dataset_root, str(line.split()[1]))
                    img_file = self.dataset_root + str(line.split()[0])
                    mask_file = self.dataset_root + str(line.split()[1])
                    self.images_train_path.append(img_file)
                    self.masks_train_path.append(mask_file)
        # Sanity check
        for img, msk in zip(self.images_train, self.masks_train):
            assert(img.shape[0] == msk.shape[0] and img.shape[1] == msk.shape[1])
        self.images_train_path = np.array(self.images_train_path)
        self.masks_train_path = np.array(self.masks_train_path)

    def _load_testing_data(self, test_list):
        """Load test images (or path to them) from disk
        """
        # if test_list is None:
        #     self.images_test, self.images_test_path = None, None
        #     return
        if not isinstance(test_list, list) and test_list:
            with open(test_list) as t:
                test_paths = t.readlines()
        elif isinstance(test_list, list):
            test_paths = test_list
        else:
            test_paths = []

        self.images_test = []
        self.images_test_path = []
        for idx, line in enumerate(test_paths):
            if self.options['in_memory']:
                # img = Image.open(os.path.join(self.dataset_root, str(line.split()[0])))
                img = Image.open(os.path.join(self.dataset_root + str(line.split()[0])))
                self.images_test.append(np.array(img, dtype=np.uint8))
                if (idx + 1) % 1000 == 0:
                    print('Loaded ' + str(idx) + ' test images')
            # self.images_test_path.append(os.path.join(self.dataset_root, str(line.split()[0])))
            self.images_test_path.append(os.path.join(self.dataset_root + str(line.split()[0])))

    ###
    ### Optical Flow and Bounding Box Helpers
    ###
    def _use_optical_flows(self):
        """Compute optical flows, use them to warp masks, save the flows' magnitudes
        """
        print('Computing optical flows and warpping masks...')
        use_warped_masks = self.options['use_warped_masks']
        with tqdm(total=len(self.videos), desc="video", ncols=80) as pbar_videos:
            # video_cnt = 0
            for video_name, frame_count in self.videos.items():
                # pbar_videos.update(video_cnt)
                assert (frame_count > 1)

                with tqdm(total=frame_count, desc="frame", ncols=80) as pbar_frames:
                    # frame_cnt = 0
                    # pbar_frames.update(frame_cnt)
                    # First frame is special case (there's no previous frame)
                    # We still call the optical flow manager so it can return an empty flow frame of the right shape
                    cur_frame_idx = self.video_frame_idx[video_name + '_0']
                    cur = self.images_train[cur_frame_idx]
                    height, width = cur.shape[0], cur.shape[1]
                    nxt = self.images_train[self.video_frame_idx[video_name + '_1']]
                    f_flow, f_flow_norm = self.optflow.compute_flow_and_norm(cur, nxt)
                    # We still need to generate empty flows and norms
                    b_flow, b_flow_norm = self.optflow.no_flow_and_norm(cur)
                    # print(b_flow_norm.shape, b_flow_norm.dtype, f_flow_norm.shape, f_flow_norm.dtype)
                    flow_norms = np.concatenate((b_flow_norm, f_flow_norm), axis=-1)
                    # print(stacked_flows.shape, stacked_flows.dtype)
                    self.flow_norms_train.append(flow_norms)
                    assert(height == flow_norms.shape[0] and width == flow_norms.shape[1] and flow_norms.shape[2] == 1)
                    # Warp masks, if requested
                    if use_warped_masks:
                        # First frame has no previous frame to wrap from, so just use current mask
                        # print("appending {} {} {} to warped masks".format(type(self.masks_train[cur_frame_idx]), self.masks_train[cur_frame_idx].shape, self.masks_train[cur_frame_idx].dtype))
                        warped_mask = self.masks_train[cur_frame_idx]
                        self.warped_prev_masks_train.append(warped_mask)
                        assert(height == warped_mask.shape[0] and width == warped_mask.shape[1] and warped_mask.shape[2] == 1)
                    # frame_cnt += 1
                    # pbar_frames.update(frame_cnt)
                    pbar_frames.update(1)

                    # Other frames have a previous and next frame
                    for frame in range(1, frame_count - 1):
                        b_flow, b_flow_norm = f_flow, f_flow_norm # already computed during the last round!
                        cur = self.images_train[self.video_frame_idx[video_name + '_' + str(frame)]]
                        # height, width = cur.shape[0], cur.shape[1]
                        nxt = self.images_train[self.video_frame_idx[video_name + '_' + str(frame + 1)]]
                        f_flow, f_flow_norm = self.optflow.compute_flow_and_norm(cur, nxt)
                        flow_norms = np.concatenate((b_flow_norm, f_flow_norm), axis=-1)
                        self.flow_norms_train.append(flow_norms)
                        assert (height == flow_norms.shape[0] and width == flow_norms.shape[1] and flow_norms.shape[2] == 1)
                        # assert (height == stacked_flows.shape[0] and width == stacked_flows.shape[1])
                        # Warp masks, if requested
                        if use_warped_masks:
                            prev_mask = self.masks_train[self.video_frame_idx[video_name + '_' + str(frame - 1)]]
                            warped_mask = self.optflow.warp_with_flow(prev_mask, b_flow)
                            # print("appending {} {} {} to warped masks".format(type(warped_mask), warped_mask.shape, warped_mask.dtype))
                            self.warped_prev_masks_train.append(warped_mask)
                            assert(height == warped_mask.shape[0] and width == warped_mask.shape[1] and warped_mask.shape[2] == 1)
                        # frame_cnt += 1
                        # pbar_frames.update(frame_cnt)
                        pbar_frames.update(1)

                    # Last frame is special case (there's no next frame)
                    b_flow, b_flow_norm = f_flow, f_flow_norm  # already computed during the last round!
                    # We still need to generate empty forward flow norm
                    cur = self.images_train[self.video_frame_idx[video_name + '_' + str(frame_count - 1)]]
                    # height, width = cur.shape[0], cur.shape[1]
                    _, f_flow_norm = self.optflow.no_flow_and_norm(cur)
                    flow_norms = np.concatenate((b_flow_norm, f_flow_norm), axis=-1)
                    self.flow_norms_train.append(flow_norms)
                    assert (height == flow_norms.shape[0] and width == flow_norms.shape[1] and flow_norms.shape[2] == 1)
                    if use_warped_masks:
                        prev_mask = self.masks_train[self.video_frame_idx[video_name + '_' + str(frame_count - 2)]]
                        warped_mask = self.optflow.warp_with_flow(prev_mask, b_flow)
                        # print("appending {} {} {} to warped masks".format(type(warped_mask), warped_mask.shape, warped_mask.dtype))
                        self.warped_prev_masks_train.append(warped_mask)
                        assert (height == warped_mask.shape[0] and width == warped_mask.shape[1] and warped_mask.shape[2] == 1)
                    # frame_cnt += 1
                    # pbar_frames.update(frame_cnt)
                    pbar_frames.update(1)
                    # video_cnt += 1
                    pbar_videos.update(1) # pbar_videos.update(video_cnt)

        # Sanity check after all videos
        if use_warped_masks:
            for img, warp in zip(self.images_train, self.warped_prev_masks_train):
                assert(img.shape[0] == warp.shape[0] and img.shape[1] == warp.shape[1])
                assert(warp.shape[2] == 1 and img.shape[2] == 3)
        for img, msk, norm in zip(self.images_train, self.masks_train, self.flow_norms_train):
            assert(img.shape[0] == msk.shape[0] and msk.shape[1] == norm.shape[1])
            assert(img.shape[0] == norm.shape[0] and img.shape[1] == norm.shape[1])
            assert (msk.shape[2] == 1 and norm.shape[2] == 1)
        print('...done with computing optical flows and warpping masks.')

    def _use_mask_bboxes(self):
        """Compute bounding boxes of masks/warped masks
        """
        for mask, warped_prev_mask in zip(self.masks_train, self.warped_prev_masks_train):
            mask_bbox = bboxes.extract_bbox(mask)
            warped_prev_mask_bbox = bboxes.extract_bbox(warped_prev_mask)
            self.masks_bboxes_train.append(np.stack((mask_bbox, warped_prev_mask_bbox), axis=-1))

    ###
    ### Batch Management
    ###
    def next_batch(self, batch_size, phase, segnet_stream='fstream'):
        """Get next batch of image (path) and masks
        Args:
            batch_size: Size of the batch
            phase: Possible options:'train' or 'test'
            segnet_stream: Binary segmentation net stream ("appearance stream" or "flow stream") ['astream'|'fstream']
        Returns in training:
            images: List of images paths if in_memory=False, List of Numpy arrays of the images if in_memory=True
            masks: List of masks paths if in_memory=False, List of Numpy arrays of the masks if in_memory=True
        Returns in testing:
            images: None if in_memory=False, Numpy array of the image if in_memory=True
            path: List of image paths
        """
        assert(self.options['in_memory']) # Only option supported at this point
        assert(segnet_stream in ['astream','fstream'])
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                if self.options['in_memory']:
                    images = []
                    if segnet_stream is 'astream':
                        for l in idx:
                            assert (self.images_train[l].shape[:2] == self.warped_prev_masks_train[l].shape[:2])
                            assert (self.images_train[l].shape[2] == 3 and self.warped_prev_masks_train[l].shape[2] == 1)
                            # print(self.images_train[l].shape, self.images_train[l].dtype)
                            # print(self.warped_prev_masks_train[l].shape, self.warped_prev_masks_train[l].dtype)
                            images.append(np.concatenate((self.images_train[l], self.warped_prev_masks_train[l]), axis=-1))
                            assert (images[len(images)-1].shape[:2] == self.images_train[l].shape[:2] and images[len(images)-1].shape[2] == 4)
                    else:
                        for l in idx:
                            # print(self.flow_norms_train[l].shape, self.flow_norms_train[l].dtype)
                            # print(self.warped_prev_masks_train[l].shape, self.warped_prev_masks_train[l].dtype)
                            assert (self.flow_norms_train[l].shape[:2] == self.warped_prev_masks_train[l].shape[:2])
                            assert (self.flow_norms_train[l].shape[2] == 2 and self.warped_prev_masks_train[l].shape[2] == 1)
                            images.append(np.concatenate((self.flow_norms_train[l], self.warped_prev_masks_train[l]), axis=-1))
                            assert (images[len(images)-1].shape[:2] == self.images_train[l].shape[:2] and images[len(images)-1].shape[2] == 3)
                    masks = [self.masks_train[l] for l in idx]
                else:
                    images = [self.images_train_path[l] for l in idx]
                    masks = [self.masks_train_path[l] for l in idx]
                self.train_ptr += batch_size
            else:
                old_idx = np.array(self.train_idx[self.train_ptr:])
                np.random.shuffle(self.train_idx)
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                idx = np.array(self.train_idx[:new_ptr])
                if self.options['in_memory']:
                    images_1 = []
                    images_2 = []
                    if segnet_stream is 'astream':
                        for l in old_idx:
                            assert (self.images_train[l].shape[:2] == self.warped_prev_masks_train[l].shape[:2])
                            assert (self.images_train[l].shape[2] == 3 and self.warped_prev_masks_train[l].shape[2] == 1)
                            images_1.append(np.concatenate((self.images_train[l], self.warped_prev_masks_train[l]), axis=-1))
                            assert (images_1[len(images_1)-1].shape[:2] == self.images_train[l].shape[:2] and images_1[len(images_1)-1].shape[2] == 4)
                        for l in idx:
                            assert (self.images_train[l].shape[:2] == self.warped_prev_masks_train[l].shape[:2])
                            assert (self.images_train[l].shape[2] == 3 and self.warped_prev_masks_train[l].shape[2] == 1)
                            images_2.append(np.concatenate((self.images_train[l], self.warped_prev_masks_train[l]), axis=-1))
                            assert (images_2[len(images_2)-1].shape[:2] == self.images_train[l].shape[:2] and images_2[len(images_2)-1].shape[2] == 4)
                    else:
                        for l in old_idx:
                            assert (self.flow_norms_train[l].shape[:2] == self.warped_prev_masks_train[l].shape[:2])
                            assert (self.flow_norms_train[l].shape[2] == 2 and self.warped_prev_masks_train[l].shape[2] == 1)
                            images_1.append(np.concatenate((self.flow_norms_train[l], self.warped_prev_masks_train[l]), axis=-1))
                            assert (images_1[len(images_1)-1].shape[:2] == self.images_train[l].shape[:2] and images_1[len(images_1)-1].shape[2] == 3)
                        for l in idx:
                            assert (self.flow_norms_train[l].shape[:2] == self.warped_prev_masks_train[l].shape[:2])
                            assert (self.flow_norms_train[l].shape[2] == 2 and self.warped_prev_masks_train[l].shape[2] == 1)
                            images_2.append(np.concatenate((self.flow_norms_train[l], self.warped_prev_masks_train[l]), axis=-1))
                            assert (images_2[len(images_2)-1].shape[:2] == self.images_train[l].shape[:2] and images_2[len(images_2)-1].shape[2] == 3)
                    masks_1 = [self.masks_train[l] for l in old_idx]
                    masks_2 = [self.masks_train[l] for l in idx]
                else:
                    images_1 = [self.images_train_path[l] for l in old_idx]
                    masks_1 = [self.masks_train_path[l] for l in old_idx]
                    images_2 = [self.images_train_path[l] for l in idx]
                    masks_2 = [self.masks_train_path[l] for l in idx]
                images = images_1 + images_2
                masks = masks_1 + masks_2
                self.train_ptr = new_ptr
            return images, masks
        elif phase == 'test':
            images = None
            if self.test_ptr + batch_size < self.test_size:
                if self.options['in_memory']:
                    images = self.images_test[self.test_ptr:self.test_ptr + batch_size]
                paths = self.images_test_path[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                if self.options['in_memory']:
                    images = self.images_test[self.test_ptr:] + self.images_test[:new_ptr]
                paths = self.images_test_path[self.test_ptr:] + self.images_test_path[:new_ptr]
                self.test_ptr = new_ptr
            return images, paths
        else:
            return None, None

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def train_img_size(self):
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width

    ###
    ### Debug utils
    ###
    def _fix_broken_flows(self):
        fixed_flow_norms_train = []
        for broken_flow_norm in self.flow_norms_train:
            flow_1 = broken_flow_norm[:, :, :, 0]
            flow_2 = broken_flow_norm[:, :, :, 1]
            fixed_flow = np.concatenate((flow_1, flow_2), axis=-1).astype(np.float32)
            fixed_flow_norms_train.append(fixed_flow)
        np.save(self.flow_norms_train_cache, fixed_flow_norms_train)

    def print_config(self):
        """Display configuration values."""
        print("\nConfiguration:")
        for k, v in self.options.items():
            print("  {:20} {}".format(k, v))

    def _test_debug(self):
        self.print_config()
        # Choose a few videos to inspect (includes data augmented videos) at random
        subset = 3
        random.seed(1969)
        videos = random.sample(list(self.videos.keys()), 3)
        print('Chosen videos: {} out of {} videos'.format(videos, self.num_videos))

        # Show the individual video frames with groundtruth masks (red) and computed mask bounding boxes (yellow)
        frames_with_gt_masks = {}
        frames_with_warped_masks = {}
        for video in videos:
            print('{}:'.format(video))
            frames_with_gt_masks[video] = self.create_frames_with_overlays(video, mode='gt')
            frames_with_warped_masks[video] = self.create_frames_with_overlays(video, mode='warped_masks')

        # Show the individual video frames with overlays
        for video in videos:
            print('{} (groundtruth):'.format(video))
            self.show_frames(frames_with_gt_masks[video])
            print('{} (warped masks from flow):'.format(video))
            self.show_frames(frames_with_warped_masks[video])

        # Same as above, but this time turn the images into a MP4 video clip
        for video in videos:
            clip = 'E:/datasets/davis2016/clips/' + video + '_gt.mp4'
            print('{} (groundtruth): Saving to {}'.format(video, clip))
            self.make_clip(clip, frames_with_gt_masks[video])
            clip = 'E:/datasets/davis2016/clips/' + video + '_warped_masks.mp4'
            print('{} (warped masks from flow): Saving to {}'.format(video, clip))
            self.make_clip(clip, frames_with_gt_masks[video])

        # Display video clips
        for video in videos:
            clip = 'E:/datasets/davis2016/clips/' + video + '_gt.mp4'
            print('{} (groundtruth): Displaying {}'.format(video, clip))
            self.show_clip(clip)
            clip = 'E:/datasets/davis2016/clips/' + video + '_warped_masks.mp4'
            print('{} (warped masks from flow): Displaying {}'.format(video, clip))
            self.show_clip(clip)

    def create_frames_with_overlays(self, video, mode='gt'):
        """Build list of individual frames in a video with masks overlayed."""
        # Overlay masks on top of images
        frames_with_overlays = []
        num_frames = self.videos[video]
        alpha = 0.6
        if mode == 'gt' or mode == 'gt_warped':
            masks = self.masks_train
            overlay_color = (255, 0, 0) # red
            bbox_color = (0, 255, 255) # cyan
        else:
            masks = self.warped_prev_masks_train
            overlay_color = (0, 255, 0)  # green
            bbox_color = (255, 255, 0)  # yellow
        for frame_number in range(num_frames):
            frame_idx = self.video_frame_idx[video + '_' + str(frame_number)]
            frame_with_overlay = visualize.apply_mask(self.images_train[frame_idx], masks[frame_idx], overlay_color, alpha)
            bbox = bboxes.extract_bbox(masks[frame_idx])
            frame_with_overlay = visualize.draw_box(frame_with_overlay, bbox, bbox_color) # y1, x1, y2, x2 order
            frames_with_overlays.append(frame_with_overlay)
        if mode == 'gt_warped':
            masks = self.warped_prev_masks_train
            overlay_color = (0, 255, 0)  # green
            bbox_color = (255, 255, 0)  # yellow
            for frame_number in range(num_frames):
                frame_idx = self.video_frame_idx[video + '_' + str(frame_number)]
                frame_with_overlay = visualize.apply_mask(frames_with_overlays[frame_number], masks[frame_idx], overlay_color, alpha, in_place = True)
                bbox = bboxes.extract_bbox(masks[frame_idx])
                frame_with_overlay = visualize.draw_box(frame_with_overlay, bbox, bbox_color) # y1, x1, y2, x2 order
                frames_with_overlays[frame_number] = frame_with_overlay
        return frames_with_overlays

    def show_frames(self, frames_with_overlays, title=None):
        """Display video frames individually."""
        visualize.display_images(frames_with_overlays)

    def make_clip(self, video_clip, frames_with_overlays):
        """Turn video into an MP4 clip.
        # Needs ffmpeg exe on Windows. You can obtain it with either:
        #  - install using conda: conda install ffmpeg -c conda-forge
        #  - download by calling: imageio.plugins.ffmpeg.download()
        """
        imageio.mimwrite(video_clip, np.array(frames_with_overlays), fps=30)

    def show_clip(self, video_clip):
        # Display video
        video = io.open(video_clip, 'r+b').read()
        encoded = base64.b64encode(video)
        return HTML(data='''<video alt="test" controls>
                        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                     </video>'''.format(encoded.decode('ascii')))