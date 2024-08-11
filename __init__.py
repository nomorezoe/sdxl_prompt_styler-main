from .sdxl_prompt_styler import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .utility_nodes import NODE_CLASS_MAPPINGS_2, NODE_DISPLAY_NAME_MAPPINGS_2
from .post_processing_nodes import NODE_CLASS_MAPPINGS_3
from .latent_util import NODE_CLASS_MAPPINGS_4
from .reactor_extends import NODE_CLASS_MAPPINGS_5, NODE_DISPLAY_NAME_MAPPINGS_5

NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_2)
NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_2)

NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_3)
NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_4)
NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_5)

NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_5)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
