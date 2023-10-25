"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from typing import Iterator

from onnx import AttributeProto

from . import Node, register


@register
class SpaceToDepth(Node):
    """https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html"""

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    @property
    def attributes(self) -> Iterator[AttributeProto]:
        blocksize = AttributeProto(
            name="blocksize", type=AttributeProto.INT, i=int(self.block_size)
        )
        if hasattr(self, "mode") and self.mode != "blocks_first":
            raise NotImplementedError(f"mode {self.mode} is not supported!")
        return (blocksize,)


@register
class DepthToSpace(Node):
    """https://onnx.ai/onnx/operators/onnx__DepthToSpace.html"""

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    @property
    def attributes(self) -> Iterator[AttributeProto]:
        blocksize = AttributeProto(
            name="blocksize", type=AttributeProto.INT, i=int(self.block_size)
        )
        mode = AttributeProto(name="mode", type=AttributeProto.STRING, s="DCR".encode())
        if hasattr(self, "mode"):
            if self.mode != "blocks_first":
                mode = AttributeProto(
                    name="mode", type=AttributeProto.STRING, s="CRD".encode()
                )
        return blocksize, mode
