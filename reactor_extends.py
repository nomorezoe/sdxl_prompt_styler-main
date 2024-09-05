import sys
import os
import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../comfyui-reactor-node"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../comfyui-reactor-node/scripts"))

from reactor_utils import tensor_to_pil
from reactor_swapper import analyze_faces

class ReActorUpdateCountFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),               
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, image):
        image = tensor_to_pil(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        det_size=(640, 640)
        faces = analyze_faces(image, det_size)
        return (len(faces), )
     
NODE_CLASS_MAPPINGS_5 = {
    "ReActorUpdateCountFace": ReActorUpdateCountFace
}

NODE_DISPLAY_NAME_MAPPINGS_5 = {
    "ReActorUpdateCountFace": "Extends ReActor Update CountFace"
}

