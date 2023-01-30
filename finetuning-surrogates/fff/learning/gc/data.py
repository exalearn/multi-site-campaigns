import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Callable, Union

from ase.db import connect
from ase import Atoms
import torch
from torch.nn import functional as F
from torch_geometric.data import InMemoryDataset, Data

default_types = {'H': 0, 'O': 1}


def convert_atoms_to_pyg(mol: Atoms, types: Optional[dict[str, int]] = None) -> Data:
    """Convert an Atoms object to a PyG Data object

    Args:
        mol: Atoms object to be converted
        types: Mapping of element to type in the network. Default is for O->0, H->1
    Returns:
        Converted object ready to be used in a PyG SchNet model
    """

    # Use default types if non provided
    if types is None:
        types = default_types

    # Center the cluster around 0
    pos = mol.get_positions() - mol.get_center_of_mass()
    pos = torch.tensor(pos, dtype=torch.float)  # coordinates

    # Only operate if there is a calculator
    y = torch.zeros((), dtype=torch.float)  # potential energy
    f = torch.zeros_like(pos)
    if mol.calc is not None:
        results = mol.calc.results
        if 'energy' in results:
            y = torch.tensor(mol.get_potential_energy(), dtype=torch.float)  # potential energy
        if 'forces' in results:
            f = torch.tensor(mol.get_forces(), dtype=torch.float)  # gradients

    # Convert it to a PyG data record
    size = int(pos.size(dim=0) / 3)
    type_idx = [types.get(i) for i in mol.get_chemical_symbols()]
    atomic_number = mol.get_atomic_numbers()
    z = torch.tensor(atomic_number, dtype=torch.long)
    x = F.one_hot(torch.tensor(type_idx, dtype=torch.long), num_classes=len(types))
    return Data(x=x, z=z, pos=pos, y=y, f=f, n_atoms=len(mol), size=size)


# TODO (wardlt): Write diskless version
class AtomsDataset(InMemoryDataset):
    """Dataset created from a list of """

    def __init__(self,
                 db_path: Union[str, Path],
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                 ):
        """
        Args:

            root: Directory where processed output will be saved
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply to data
            pre_filter: Pre-filter to apply to data
        """
        self.db_path = Path(db_path).absolute()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # NB: database file
        return [self.db_path.name]

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        # Keep a record of the raw data along with the processed results
        shutil.copyfile(self.db_path, self.raw_paths[0])

    def process(self):
        # NB: coding for atom types
        data_list = []

        # Loop over all datasets (there should be only one)
        for db_path in self.raw_paths:
            with connect(db_path) as conn:
                # Loop over all records
                for row in conn.select(''):
                    # Extract the data from the ASE records
                    mol: Atoms = row.toatoms()
                    data = convert_atoms_to_pyg(mol)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                        # The graph edge_attr and edge_indices are created when the transforms are applied
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        return self.collate(data_list)

    @classmethod
    def from_atoms(cls, atoms_lst: list[Atoms], root: Union[str, Path], **kwargs) -> 'AtomsDataset':
        """Make a data loader from a list of ASE atoms objects

        Args:
            atoms_lst: List of atoms objects to put in dataset
            root: Directory to store the cached data
        """

        with TemporaryDirectory() as tmp:
            # Write a database that will get deleted after you make the loader
            db_path = Path(tmp) / 'temp.db'
            with connect(db_path, type='db') as conn:
                for atoms in atoms_lst:
                    conn.write(atoms)

            # Make the loader
            return cls(db_path, root=root, **kwargs)
