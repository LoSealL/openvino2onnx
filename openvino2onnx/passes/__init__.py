"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import inspect
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Type

from tabulate import tabulate

from .auto_load import auto_load
from .rewriter import Rewriter


class Registry:
    """A simple registry object to hold objects from others

    Samples::

        FOO = Registry("FOO")

        @FOO.register()
        def foo(): ...

        print(FOO)
        # ┌───────────────┐
        # │ Register: FOO │
        # ├───────────────┤
        # │ foo           │
        # └───────────────┘
    """

    class _FuncWrapper:
        def __init__(self, func):
            self.func = func
            self.__deps__ = func.__deps__
            self.__patches__ = func.__patches__
            self.__name__ = func.__name__

        def __call__(self):
            return self.func

    def __init__(self, name=None, parent: Optional["Registry"] = None) -> None:
        self._bucks = {}
        self._configs = {}
        self._name = name or "<Registry>"
        self._parent = parent
        if parent is not None:
            self._name = f"{parent.name}.{self.name}"

    @property
    def name(self) -> str:
        """Return the name of the registry."""
        return self._name

    @staticmethod
    def _legal_name(name: str) -> str:
        words = [""]
        for a, b in zip(list(name), list(name.lower())):
            if a != b:
                words.append("")
            words[-1] += b
        return "_".join(words).strip("_")

    def register(
        self,
        name=None,
        deps: Optional[List[str]] = None,
        patch: Optional[List[str]] = None,
    ):
        """A decorator to register an object.

        Args:
            name (str, optional): The name of the object. If not provided, the name
                of the function of class will be used after transform to lowercase.
            deps (List[str], optional): The dependencies before executing the object.
            patch (List[str], optional): The hook after the object execution.
        """

        def wrapper(func: Callable | Type[Rewriter]):
            if not callable(func):
                raise TypeError(
                    "the object to be registered must be a function or Rewriter,"
                    f" got {type(func)}"
                )
            # set default dependency list to empty
            setattr(func, "__deps__", deps or [])
            setattr(func, "__patches__", patch or [])
            if inspect.isfunction(func):
                func_wrap = self._FuncWrapper(func)
                func_wrap.__name__ = name or func.__name__
                self._bucks[func_wrap.__name__] = func_wrap
                self._configs[func_wrap.__name__] = inspect.signature(func)
            else:
                obj: Rewriter = func()  # type: ignore
                if not (hasattr(obj, "rewrite") and inspect.ismethod(obj.rewrite)):
                    raise TypeError(
                        f"the registered object {func} must be the subclass "
                        "of openvino2onnx.passes.rewriter.Rewriter, but its mro is "
                        f"{func.__mro__}"  # type: ignore
                    )
                assert callable(obj), f"Wired! {func} is not callable!"

                # note name is not saved because obj is gc-ed after this function
                obj.__name__ = name or self._legal_name(func.__name__)
                self._bucks[obj.__name__] = func
                self._configs[obj.__name__] = inspect.signature(func.rewrite)
            if self._parent is not None:
                self._parent.register(name, deps, patch)(func)
            return func

        return wrapper

    def get(self, name: str) -> Optional[Callable | Rewriter]:
        """Get a registered object by its name."""
        if name in self._bucks:
            functor = self._bucks[name]()  # create a new instance each time
            functor.__name__ = name  # rename the instance
            return functor

    def get_config(self, name: str):
        """Get the configuration of an object"""
        return self._configs.get(name)

    def child(self, passes: str | Sequence[str]) -> "Registry":
        """Slice a child registry by given a set of pass names."""

        reg = self.__class__(parent=self)
        if isinstance(passes, str):
            passes = [passes]
        # pylint: disable=protected-access
        reg._bucks = {k: self._bucks[k] for k in passes}
        reg._configs = {k: self._configs[k] for k in passes}
        return reg

    def __iter__(self) -> Iterator[str]:
        """Return an Iterator for all registered functions"""
        yield from self._bucks.keys()

    def __contains__(self, name: str) -> bool:
        """Check if a function is registered"""
        return name in self._bucks

    def __repr__(self) -> str:
        title = [f"Register: {self._name}", "Deps", "Patch", "Config"]
        members = []
        for i in sorted(self._bucks.keys()):
            members.append(
                [
                    i,
                    self._bucks[i].__deps__,
                    self._bucks[i].__patches__,
                    self._configs[i],
                ]
            )
        return tabulate(members, title, "simple_grid", maxcolwidths=[None, 50, 50, 50])


PASSES = Registry("PASS")
L1 = Registry("L1", parent=PASSES)
L2 = Registry("L2", parent=PASSES)
L3 = Registry("L3", parent=PASSES)

auto_load(Path(__file__).parent / "convert")
auto_load(Path(__file__).parent / "optimize")
auto_load(Path(__file__).parent / "quantize")
auto_load(Path(__file__).parent / "transform")
