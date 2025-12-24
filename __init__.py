"""
STAR Video Super Resolution for ComfyUI
A wrapper for STAR VSR model that provides video upscaling capabilities.
"""

print(r"""   _____ _______       _____  
  / ____|__   __|/\   |  __ \ 
 | (___    | |  /  \  | |__) |
  \___ \   | | / /\ \ |  _  / 
  ____) |  | |/ ____ \| | \ \ 
 |_____/   |_/_/    \_\_|  \_\
                              """)

try:
    import xformers
except ImportError:
    raise ImportError(
        "Please install xformers manually version before use this custom nodes see instruction at https://github.com/vjumpkung/ComfyUI-STARWrapper"
    )


from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
