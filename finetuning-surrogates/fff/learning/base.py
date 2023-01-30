"""Interface definitions"""
from pathlib import Path
from typing import Union, Optional

import ase
import numpy as np
import pandas as pd
import torch.nn

from .util.messages import TorchMessage

# TODO (wardlt): Break the hard-wire to PyTorch

ModelMsgType = Union[TorchMessage, torch.nn.Module, Path]


class BaseLearnableForcefield:
    """Define the functions for learning and evaluating a forcefield"""

    def __init__(self, scratch_dir: Optional[Path] = None):
        """

        Args:
            scratch_dir: Path used to store temporary files
        """
        self.scratch_dir = scratch_dir

    def get_model(self, model_msg: ModelMsgType) -> torch.nn.Module:
        """Load a model from the provided message and place on the main memory

        May load from disk or unpack a model from memory.

        Args:
            model_msg: Model message (could be the model itself)
        Returns:
            The model unserialized
        """
        if isinstance(model_msg, TorchMessage):
            return model_msg.get_model(map_location='cpu')
        elif isinstance(model_msg, (Path, str)):
            return torch.load(model_msg, map_location='cpu')  # Load to main memory first
        elif isinstance(model_msg, torch.nn.Module):
            return model_msg
        else:
            raise ValueError(f'Unsupported message type: {type(model_msg)}')

    def evaluate(self,
                 model_msg: ModelMsgType,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (list[float], list[np.ndarray]):
        """Run inference for a series of structures

        Args:
            model_msg: Model to evaluate. Either a SchNet model or the bytes corresponding to a serialized model
            atoms: List of structures to evaluate
            batch_size: Number of molecules to evaluate per batch
            device: Device on which to run the computation
        Returns:
            - Energies for each inference
            - Forces for each inference
        """
        ...

    def train(self,
              model_msg: ModelMsgType,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              huber_deltas: (float, float) = (0.5, 1),
              energy_weight: float = 0.9,
              reset_weights: bool = False,
              patience: int = None) -> (TorchMessage, pd.DataFrame):
        """Train a model

        Args:
            model_msg: Model to be retrained
            train_data: Structures used for training
            valid_data: Structures used for validation
            num_epochs: Number of training epochs
            device: Device (e.g., 'cuda', 'cpu') used for training
            batch_size: Batch size during training
            learning_rate: Initial learning rate for optimizer
            huber_deltas: Delta parameters for the loss functions for energy and force
            energy_weight: Amount of weight to use for the energy part of the loss function
            reset_weights: Whether to reset the weights before training
            patience: Patience until learning rate is lowered. Default: epochs / 8
        Returns:
            - model: Retrained model
            - history: Training history
        """
        ...
