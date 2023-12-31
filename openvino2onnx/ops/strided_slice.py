"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from . import Node, register


@register
class StridedSlice(Node):
    @property
    def type_name(self):
        return "Slice"
