"""
optflow.py

Optical flow utility functions and classes.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

TODO:
- Add support for flownet2

References for future work:
    # https://github.com/sampepose/flownet2-tf/blob/master/src/flowlib.py
    # https://github.com/sampepose/flownet2-tf  <== Use this to generate the flows!
    # https://github.com/NVIDIA/flownet2-pytorch
    # https://github.com/lmb-freiburg/flownet2
    # https://docs.opencv.org/3.3.1/db/d7f/tutorial_js_lucas_kanade.html
    # https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
    # https://github.com/suyogduttjain/fusionseg/tree/master/external_libs/OpticalFlow
    # https://github.com/anchen1011/toflow/blob/master/src/util/ut_flowx.lua
    # https://github.com/anuragranj/spynet#flowUtils
    # https://github.com/anuragranj/spynet/blob/master/flowExtensions.lua
    # https://github.com/marcoscoffier/optical-flow/blob/master/init.lua
    # https://www.mathworks.com/matlabcentral/answers/23708-using-optical-flow-to-warp-an-image?requestedDomain=true
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys

import pyflow

_DEFAULT_OPTLOW_OPTIONS = {
    'optical_flow_mgr': 'pyflow',
    'pyflow_alpha' :  0.012,
    'pyflow_ratio': 0.75,
    'pyflow_minWidth': 20,
    'pyflow_nOuterFPIterations': 7,
    'pyflow_nInnerFPIterations': 1,
    'pyflow_nSORIterations': 30,
    'pyflow_colType': 0, # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    'pyflow_threshold': 0.000005
}

class optflow(object):
    def __init__(self, options=_DEFAULT_OPTLOW_OPTIONS):
        """Initialize the optical flow object
        Args:
        options: Configuration options
        Options:
          use_flow_magnitues: True computes magnitudes of forward and backward flows
        Returns:
        """
        # Check settings
        self.options = options
        if not options['optical_flow_mgr'] is 'pyflow':
            raise ValueError('pyflow is the only optical flow technique supported so far!')
        else:
            # Initialize pyflow options
            self.alpha = self.options['pyflow_alpha']
            self.ratio = self.options['pyflow_ratio']
            self.minWidth = self.options['pyflow_minWidth']
            self.nOuterFPIterations = self.options['pyflow_nOuterFPIterations']
            self.nInnerFPIterations = self.options['pyflow_nInnerFPIterations']
            self.nSORIterations = self.options['pyflow_nSORIterations']
            self.colType = self.options['pyflow_colType']
            self.threshold = self.options['pyflow_threshold']

        # Build "no flow" caches
        self.no_flow_cache = {}
        self.no_flow_norm_cache = {}

    ###
    ### Public Methods
    ###
    def compute_flow_and_norm(self, src, dst):
        """Given two images, compute the src -> dst optical flow and their norms
        Args:
            src, dst: Two sequential frames (H,W,3)
        Returns:
            flow (H,W,2) and its norm (H,W,1)
        """
        delta_x, delta_y = self._compute_flow_pyflow(src, dst)
        flow_norm = np.linalg.norm((delta_x, delta_y), axis=0)
        flow = np.stack((delta_x, delta_y), axis=-1)
        flow_norm = np.expand_dims(flow_norm, axis=-1).astype(np.float32)
        assert (len(flow.shape) == 3 and flow.shape[2] == 2)
        assert (len(flow_norm.shape) == 3 and flow_norm.shape[2] == 1)
        # print(flow.shape, flow.dtype, flow_norm.shape, flow_norm.dtype)
        return flow, flow_norm

    def warp_with_flow(self, src, flow):
        """warp an image using an optical flow
        Args:
            src: Mask to warp (H,W,1) with flow (H,W,2)
        Returns:
            Warped mask (H,W,1)
        """
        return self._warp_with_flow_pyflow(src, flow)

    def no_flow_and_norm(self, img):
        """Get a null, static flow, and its zero norm
        Args:
            img: input frame  (H,W,3)
        Returns:
            null flow (H,W,2) and its zero norm (H,W,1)
        """
        key = '{}x{}'.format(img.shape[0], img.shape[1])
        if key in self.no_flow_cache:
            # print("reading no_flow buffers of size ({}) from cache".format(key))
            no_flow = self.no_flow_cache[key]
            no_flow_norm = self.no_flow_norm_cache[key]
        else:
            # print("adding no_flow buffers of size ({}) to cache".format(key))
            no_flow = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float)
            no_flow_norm = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
            self.no_flow_cache[key], self.no_flow_norm_cache[key] = no_flow, no_flow_norm

        return no_flow, no_flow_norm

    def visualize_flow(self, flow):
        """Given an optical flow, create an image easy to display
        """
        # TODO visualize_flow
        pass

    def visualize_flow_norm(self, flow_norm):
        """Given an optical flow norm, create an image easy to display
        """
        # TODO visualize_flow_norm
        pass

    ###
    ### Core Methods
    ###
    def _compute_flow_pyflow(self, img1, img2):
        """Given two images, compute the im1 -> im2 optical flow
        """
        delta_x, delta_y, _ = pyflow.coarse2fine_flow(
            img1.astype(float) / 255., img2.astype(float) / 255.,
            self.alpha, self.ratio, self.minWidth, self.nOuterFPIterations, self.nInnerFPIterations,
            self.nSORIterations, self.colType, verbose=False, threshold=self.threshold)
        assert (len(delta_x.shape) == 2 and len(delta_y.shape) == 2)
        return delta_x, delta_y

    def _warp_with_flow_pyflow(self, img, flow):
        """warp an image using an optical flow
        """
        # pyflow.warp_mask_flow() takes an (H,W,c) input with pixel values between 0. and 1.
        if img.ndim==2:
            img_dim3 = np.expand_dims(img, axis=-1).astype(float) / 255.
        else:
            img_dim3 = img.astype(float) / 255.
        delta_x, delta_y = np.ascontiguousarray(flow[:,:,0]), np.ascontiguousarray(flow[:,:,1])
        warped_img = pyflow.warp_mask_flow(img_dim3, delta_x, delta_y)
        warped_img = np.uint8(np.where(warped_img>0.5, 1., 0.) * 255.)
        if img.ndim==2:
            warped_img = warped_img[:,:,0]
        return warped_img



