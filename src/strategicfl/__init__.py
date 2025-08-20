"""
StrategicFL: A Framework for Strategic Federated Learning

A Python framework for simulating strategic federated learning scenarios where
clients can behave adversarially by manipulating gradient updates.
"""

__version__ = "0.1.0"
__author__ = "Dimitar Chakarov"

# Core imports for easy access
from .agents import Client, Server

__all__ = [
    "Client",
    "Server",
]
