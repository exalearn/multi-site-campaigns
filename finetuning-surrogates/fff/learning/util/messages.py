"""Tools for sending machine learning models between different machines"""
from io import BytesIO
from typing import Union

import torch


class TorchMessage:
    """Send a PyTorch object via pickle, enable loading on to target hardware"""

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: Model to be sent
        """
        self.model = model
        self._pickle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the model with pickle
        model_pkl = BytesIO()
        torch.save(self.model, model_pkl)

        # Store it
        state['model'] = None
        state['_pickle'] = model_pkl.getvalue()
        return state

    def get_model(self, map_location: Union[str, torch.device] = 'cpu'):
        """Load the cached model into memory

        Args:
            map_location: Where to copy the device
        Returns:
            Deserialized model, moved to the target resource
        """
        if self.model is None:
            self.model = torch.load(BytesIO(self._pickle), map_location=map_location)
            self.model.to(map_location)
            self._pickle = None
        else:
            self.model.to(map_location)
        return self.model
