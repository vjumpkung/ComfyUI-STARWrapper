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

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("🎉 Loading STAR Video Super-Resolution Completed ")
