import sys
import os
import cv2
import numpy as np
import torch
# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Add the My directory path to the sys.path list
sys.path.append(my_dir)
reactor_dir = os.path.abspath(os.path.join(my_dir, '..', 'comfyui-reactor-node'))
sys.path.append(reactor_dir)

from reactor_utils import tensor_to_pil
#from scripts.reactor_faceswap import analyze_faces

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
        #faces = analyze_faces(image, det_size)
        return 4
     
NODE_CLASS_MAPPINGS_5 = {
    "ReActorUpdateCountFace": ReActorUpdateCountFace
}

NODE_DISPLAY_NAME_MAPPINGS_5 = {
    "ReActorUpdateCountFace": "Extends ReActor Update CountFace"
}

