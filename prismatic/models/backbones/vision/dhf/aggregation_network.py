# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================================================
# This code is adapted from aggregation_network.py in Diffusion Hyperfeatures
# and implements a learned aggregation of diffusion features, 
# with additional functionality for feeding the features to an output_head.
# Original source: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures/blob/main/archs/aggregation_network.py
# =================================================================

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import einops

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        num_norm_groups = kwargs.pop("num_norm_groups")
       
        super().__init__(*args, **kwargs)
        self.norm = nn.GroupNorm(num_norm_groups, kwargs["out_channels"])

    def forward(self, x):
        x = F.conv2d(
            x, 
            self.weight, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )
        x = self.norm(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        num_norm_groups=32,
        emb_channels=1280
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                num_norm_groups=num_norm_groups
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            num_norm_groups=num_norm_groups
        )
        self.conv2 = Conv2d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=1,
            dilation=1,
            num_norm_groups=num_norm_groups
        )
        self.conv3 = Conv2d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            num_norm_groups=num_norm_groups
        )

        # Weight initialization
        if self.shortcut is not None:
            weight_init.c2_msra_fill(self.shortcut)
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)

        # Create timestep conditioning layers
        if emb_channels > 0:
            self.emb_layers = nn.Linear(emb_channels, bottleneck_channels)
        else:
            self.emb_layers = nn.Identity()

    def forward(self, x, emb=None):
        out = self.conv1(x)
        out = F.relu(out)

        # Add timestep conditioning
        if emb is not None:
            emb = emb.to(out.dtype)
            emb = emb.to(out.device)
            if emb.shape[0] > out.shape[0]:
                emb = emb[:out.shape[0]]
            emb_out = self.emb_layers(emb)
            emb_out = F.relu(emb_out)
            out = out + emb_out[:, :, None, None]

        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu(out)
        return out


def get_resampler(in_channels, out_channels, out_res, in_res=64):
    out_res = str(out_res)
    if in_res == 64:
        resamplers = {
            "32": nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            "64": nn.Identity(),
            "16": nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=4, padding=1),
            "24": nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=3, padding=4),
        }
    elif in_res == 128:
        resamplers = {
            "32": nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=4, padding=1),
            "64": nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            "16": nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=8, padding=1),
            "24": nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=5, padding=1, dilation=5),
        }
    if out_res not in resamplers:
        raise ValueError(f"Invalid output resolution: {out_res}")
    return resamplers[out_res]


class AggregationNetwork(nn.Module):
    def __init__(
            self, 
            feature_dims, 
            projection_dim=384,
            num_norm_groups=32,
            save_timestep=[0],
            bottleneck_sequential=True, # if True, don't condition on time embedding
            reshape_outputs=False,
            final_resolution=None,
            input_resolution=64
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims
        self.reshape_outputs = reshape_outputs
        self.final_resolution = final_resolution

        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = BottleneckBlock(
                in_channels=feature_dim,
                bottleneck_channels=projection_dim // 4,
                out_channels=projection_dim,
                num_norm_groups=num_norm_groups,
                emb_channels=1280 if not bottleneck_sequential else 0
            )
            if bottleneck_sequential:
                bottleneck_layer = nn.Sequential(bottleneck_layer)
            self.bottleneck_layers.append(bottleneck_layer)
        
        self.bottleneck_layers = self.bottleneck_layers
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights)
        if self.reshape_outputs: # reshape the output to the final resolution
            self.resampler = get_resampler(projection_dim, projection_dim, final_resolution, input_resolution)

    def forward(self, batch, emb=None):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        if len(batch.shape) == 3:
            h = w = int(np.sqrt(batch.shape[1])) # assume square image
            batch = einops.rearrange(batch, 'b (h w) c -> b c h w', h=h, w=w)
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights, dim=0)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            if type(bottleneck_layer) is not nn.Sequential:
                bottlenecked_feature = bottleneck_layer(feats, emb)
            else:
                bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        if self.reshape_outputs:
            output_feature = self.resampler(output_feature)
        output_feature = einops.rearrange(output_feature, 'b c w h -> b (w h) c')
        return output_feature

    def is_requires_grad(self):
        return all(p.requires_grad for p in self.parameters())

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())