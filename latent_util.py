# Copyright 2023 SLAPaper
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

import collections.abc as clabc
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import numpy as np
import sys

import os
import json
from comfy.cli_args import args
import torch
import folder_paths

class SaveImage_plus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                    "boolean": ("BOOLEAN", {"default": False}),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, boolean, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        if( boolean== False):
            return { "ui": { "images": [] } }
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        print("save_images"  + str(len(images)))
        for (batch_number, image) in enumerate(images):
            print("batch:"+ str(batch_number))
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            break

        return { "ui": { "images": results } }

class ImageSelector:
    """
    Select some of the images and pipe through
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Input: list of index of selected image, seperated by comma (",")
        support colon (":") sperated range (left included, right excluded) 
        Indexes start with 1 for simplicity
        """
        return {
            "required": {
                "images": ("IMAGE", ),
                "selected_indexes": ("STRING", {
                    "multiline": False,
                    "default": "1,2,3"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "run"

    OUTPUT_NODE = False

    CATEGORY = "image"

    def run(self, images: torch.Tensor, selected_indexes: str):
        """
        根据 selected_indexes 选择 images 中的图片，支持连续索引和范围索引

        Args:
            images (torch.Tensor): 输入的图像张量，维度为 [N, C, H, W], 其中 N 为图片数量, C 为通道数, H、W 为图片的高和宽。
            selected_indexes (str): 选择的图片索引，支持连续索引和范围索引，例如："0,2,4:6,8" 表示选择第1、3、5张和第2、4、6、8张图片。

        Returns:
            tuple: 选择的图片张量，维度为 [N', C, H, W]，其中 N' 为选择的图片数量。

        """
        shape = images.shape
        len_first_dim = shape[0]

        selected_index: list[int] = []
        total_indexes: list[int] = list(range(len_first_dim))
        for s in selected_indexes.strip().split(','):
            try:
                if ":" in s:
                    _li = s.strip().split(':', maxsplit=1)
                    _start = _li[0]
                    _end = _li[1]
                    if _start and _end:
                        selected_index.extend(
                            total_indexes[int(_start) - 1:int(_end) - 1]
                        )
                    elif _start:
                        selected_index.extend(
                            total_indexes[int(_start) - 1:]
                        )
                    elif _end:
                        selected_index.extend(
                            total_indexes[:int(_end) - 1]
                        )
                else:
                    x: int = int(s.strip()) - 1
                    if x < len_first_dim:
                        selected_index.append(x)
            except:
                pass

        if selected_index:
            print(f"ImageSelector: selected: {len(selected_index)} images")
            return (images[selected_index], )

        print(f"ImageSelector: selected no images, passthrough")
        return (images, )


class ImageDuplicator:
    """
    Duplicate each images and pipe through
    """

    def __init__(self):
        self._name = "ImageDuplicator"
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Input: copies you want to get
        """
        return {
            "required": {
                "images": ("IMAGE", ),
                "dup_times": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "run"

    OUTPUT_NODE = False

    CATEGORY = "image"

    def run(self, images: torch.Tensor, dup_times: int):
        """
        对输入的图像张量进行复制多次，并将复制后的张量拼接起来返回。

        Args:
            images (torch.Tensor): 输入的图像张量，维度为 (batch_size, channels, height, width)。
            dup_times (int): 复制的次数。

        Returns:
            torch.Tensor: 拼接后的图像张量，维度为 (batch_size * dup_times, channels, height, width)。

        """

        tensor_list = [images
                       ] + [torch.clone(images) for _ in range(dup_times - 1)]

        print(
            f"ImageDuplicator: dup {dup_times} times,",
            f"return {len(tensor_list)} images",
        )
        return (torch.cat(tensor_list), )


class LatentSelector:
    """
    Select some of the latent images and pipe through
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Input: list of index of selected image, seperated by comma (",")
        support colon (":") sperated range (left included, right excluded) 
        Indexes start with 1 for simplicity
        """
        return {
            "required": {
                "latent_image": ("LATENT", ),
                "selected_indexes": ("STRING", {
                    "multiline": False,
                    "default": "1,2,3"
                }),
            },
        }

    RETURN_TYPES = ("LATENT", )
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "run"

    OUTPUT_NODE = False

    CATEGORY = "latent"

    def run(self, latent_image: clabc.Mapping[str, torch.Tensor],
            selected_indexes: str):
        """
        对latent_image进行筛选，根据selected_indexes指定的索引进行筛选
        Args:
            latent_image: 待筛选的latent_image，Mapping[str, torch.Tensor]，包含'samples'字段
            selected_indexes: 待筛选的索引，以逗号分隔，支持连续索引范围以冒号分隔，例如'1,3,5:7,9'

        Returns:
            筛选后的latent_image，Mapping[str, torch.Tensor]
        """
        samples = latent_image['samples']
        shape = samples.shape
        len_first_dim = shape[0]

        selected_index: list[int] = []
        total_indexes: list[int] = list(range(len_first_dim))
        for s in selected_indexes.strip().split(','):
            try:
                if ":" in s:
                    _li = s.strip().split(':', maxsplit=1)
                    _start = _li[0]
                    _end = _li[1]
                    if _start and _end:
                        selected_index.extend(
                            total_indexes[int(_start) - 1:int(_end) - 1]
                        )
                    elif _start:
                        selected_index.extend(
                            total_indexes[int(_start) - 1:]
                        )
                    elif _end:
                        selected_index.extend(
                            total_indexes[:int(_end) - 1]
                        )
                else:
                    x: int = int(s.strip()) - 1
                    if x < len_first_dim:
                        selected_index.append(x)
            except:
                pass

        if selected_index:
            print(f"LatentSelector: selected: {len(selected_index)} latents")
            return ({'samples': samples[selected_index, :, :, :]}, )

        print(f"LatentSelector: selected no latents, passthrough")
        return (latent_image, )

class LatentDuplicator:
    """
    Duplicate each latent images and pipe through
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Input: copies you want to get
        """
        return {
            "required": {
                "latent_image": ("LATENT", ),
                "dup_times": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("LATENT", )
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "run"

    OUTPUT_NODE = False

    CATEGORY = "latent"

    def run(self, latent_image: clabc.Mapping[str, torch.Tensor],
            dup_times: int):
        """
        对latent_image进行复制, 复制次数为dup_times。
        
        Args:
            latent_image (clabc.Mapping[str, torch.Tensor]): 输入的latent_image, 包含'samples'键。
            dup_times (int): 复制次数。
        
        Returns:
            Tuple[Dict[str, torch.Tensor]]: 返回包含samples的字典, samples是一个长度为(dup_times+1)的样本张量。
        
        """
        samples = latent_image['samples']

        sample_list = [samples] + [
            torch.clone(samples) for _ in range(dup_times - 1)
        ]

        print(
            f"LatentDuplicator: dup {dup_times} times,",
            f"return {len(sample_list)} images",
        )
        return ({
            'samples': torch.cat(sample_list),
        }, )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS_4 = {
    "ImageSelector": ImageSelector,
    "ImageDuplicator": ImageDuplicator,
    "LatentSelector": LatentSelector,
    "LatentDuplicator": LatentDuplicator,
    "SaveImage_plus": SaveImage_plus
}