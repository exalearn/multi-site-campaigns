"""ASE interface to the model"""
from io import BytesIO
from typing import Optional

import torch
from ase.calculators.calculator import Calculator, all_changes

from fff.learning.gc.functions import eval_batch
from fff.learning.gc.models import SchNet
from fff.learning.gc.data import convert_atoms_to_pyg


class SchnetCalculator(Calculator):
    """ASE interface to trained model

    Args:
        model: SchNet
    """

    implemented_properties = ['forces', 'energy']
    nolabel = True

    def __init__(self, model: SchNet, device: Optional[str] = None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.net = model
        self.device = device

    def __getstate__(self):
        # Remove the model from the serialization dictionary
        state = self.__dict__.copy()
        net = state.pop('net')

        # Serialize it using Torch's "save" functionality
        fp = BytesIO()
        torch.save(net, fp)
        state['_net'] = fp.getvalue()
        return state

    def __setstate__(self, state):
        # Load it back in
        fp = BytesIO(state.pop('_net'))
        state['net'] = torch.load(fp, map_location='cpu')
        self.__dict__ = state

    def to(self, device: str):
        """Move the model to a certain device"""
        self.net.to(device)
        self.device = device

    def calculate(
            self, atoms=None, properties=None, system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        if atoms is not None:
            self.atoms = atoms.copy()

        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert the atoms object to a PyG Data
        data = convert_atoms_to_pyg(atoms)
        data.batch = torch.zeros((len(atoms)))  # Fake a single-item batch

        # Run the "batch"
        if self.device is not None:
            data.to(self.device)
            self.net.to(self.device)
        energy, gradients = eval_batch(self.net, data)
        self.results['energy'] = energy.item()
        self.results['forces'] = gradients.cpu().detach().numpy().squeeze()
