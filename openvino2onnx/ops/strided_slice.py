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

    @property
    def attributes(self):
        if self.new_axis_mask or self.shrink_axis_mask or self.ellipsis_mask:
            raise NotImplementedError(
                "expect new_axis_mask, shrink_axis_mask, and ellipsis_mask to be empty"
            )
        return []
