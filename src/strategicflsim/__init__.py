"""
strategic-fl-sim: A Package for Strategic Federated Learning

A Python package for simulating strategic federated learning scenarios where
clients can behave adversarially by manipulating gradient updates.
"""

__version__ = "0.1.0"

# Core imports for easy access
from .agents import Client, Server

__all__ = [
    "Client",
    "Server",
]
