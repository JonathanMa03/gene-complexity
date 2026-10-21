"""Reproducible single-cell representation comparison."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "Workflow":
        from .workflow import Workflow
        return Workflow
    raise AttributeError(name)
