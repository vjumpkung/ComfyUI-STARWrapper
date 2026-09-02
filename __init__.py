"""STAR Video Super Resolution nodes for ComfyUI."""

from comfy_api.latest import ComfyExtension, io

from .nodes import STAR_NODE_CLASSES


class STARVSRWrapperExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return STAR_NODE_CLASSES


async def comfy_entrypoint() -> STARVSRWrapperExtension:
    return STARVSRWrapperExtension()


__all__ = ["comfy_entrypoint"]
