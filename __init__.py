from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .moss_regional_lora import MOSSImageRegionalLoRA


class MOSSRegionalLoRAExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MOSSImageRegionalLoRA]


async def comfy_entrypoint() -> MOSSRegionalLoRAExtension:
    return MOSSRegionalLoRAExtension()
