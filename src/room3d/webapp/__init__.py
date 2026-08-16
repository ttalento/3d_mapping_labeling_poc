"""Local web viewer for reconstructed and labeled rooms."""

from .server import create_app, serve

__all__ = ["create_app", "serve"]
