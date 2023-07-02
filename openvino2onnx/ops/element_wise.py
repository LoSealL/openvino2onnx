"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from . import Node, register


@register
class Subtract(Node):
    @property
    def type_name(self):
        return "Sub"


@register
class Multiply(Node):
    @property
    def type_name(self):
        return "Mul"
