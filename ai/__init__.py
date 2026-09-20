"""
ai package
----------
Provides pre-trained deep learning inference for wall crack classification.
"""

from .crack_classifier import CrackClassifier, get_classifier

__all__ = ["CrackClassifier", "get_classifier"]
