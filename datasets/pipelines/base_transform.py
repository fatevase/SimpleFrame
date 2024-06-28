
"""
from abc import ABC, abstractmethod
from utils import ConsoleLogger
"""
import abc
import random

class BaseTransform(abc.ABC):
    """BaseTrasnform class"""

    def __init__(self, p):
        super().__init__()
        self.logger = None

    @abc.abstractmethod
    def transform(self, x:dict):
        """Transform function
        Arguments:
            x {dict} -- frame data
        """
        raise NotImplementedError

    def __call__(self, data:dict):
        """Perform transformation
        Arguments:
            data {dict} -- frame data
        """
        return self.transform(data)