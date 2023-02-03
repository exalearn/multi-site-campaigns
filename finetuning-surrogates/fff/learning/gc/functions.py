"""Functions that use the model through interfaces designed for workflow engines"""
import os
import time
from collections import defaultdict
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

import ase
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.data import DataLoader, Data

from fff.learning.gc.data import AtomsDataset
from fff.learning.gc.models import SchNet
from fff.learning.base import BaseLearnableForcefield, ModelMsgType
from fff.learning.util.messages import TorchMessage


def eval_batch(model: SchNet, batch: Data) -> (torch.Tensor, torch.Tensor):
    """Get the energies and forces for a certain batch of molecules

    Args:
        model: Model to evaluate
        batch: Batch of data to evaluate
    Returns:
        Energy and forces for the batch
    """
    batch.pos.requires_grad = True
    energ_batch = model(batch)
    force_batch = -torch.autograd.grad(energ_batch, batch.pos, grad_outputs=torch.ones_like(energ_batch), retain_graph=True)[0]
    return energ_batch, force_batch


class GCSchNetForcefield(BaseLearnableForcefield):
    """Standardized interface to Graphcore's implementation of SchNet"""

    def evaluate(self,
                 model_msg: ModelMsgType,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> tuple[list[float], list[np.ndarray]]:
        model = self.get_model(model_msg)

        # Place the model on the GPU in eval model
        model.eval()
        model.to(device)

        with TemporaryDirectory() as tmp:
            # Make the data loader
            with open(os.devnull, 'w') as fp, redirect_stderr(fp):
                dataset = AtomsDataset.from_atoms(atoms, root=tmp)
                loader = DataLoader(dataset, batch_size=batch_size)

            # Run all entries
            energies = []
            forces = []
            for batch in loader:
                # Move the data to the array
                batch.to(device)

                # Get the energies then compute forces with autograd
                energ_batch, force_batch = eval_batch(model, batch)

                # Split the forces
                n_atoms = batch.n_atoms.cpu().detach().numpy()
                forces_np = force_batch.cpu().detach().cpu().numpy()
                forces_per = np.split(forces_np, np.cumsum(n_atoms)[:-1])

                # Add them to the output lists
                energies.extend(energ_batch.detach().cpu().numpy().tolist())
                forces.extend(forces_per)

        model.to('cpu')  # Move it off the GPU memory

        return energies, forces

    def train(self,
              model_msg: ModelMsgType,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              huber_deltas: (float, float) = (0.5, 1),
              energy_weight: float = 0.1,
              reset_weights: bool = False,
              patience: int = None) -> (TorchMessage, pd.DataFrame):

        model = self.get_model(model_msg)

        # Unpack some inputs
        huber_eng, huber_force = huber_deltas

        # If desired, re-initialize weights
        if reset_weights:
            for module in model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

        # Start the training process
        with TemporaryDirectory(prefix='spk') as td:
            td = Path(td)
            # Save the batch to an ASE Atoms database
            with open(os.devnull, 'w') as fp, redirect_stderr(fp):
                train_dataset = AtomsDataset.from_atoms(train_data, td / 'train')
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                valid_dataset = AtomsDataset.from_atoms(valid_data, td / 'valid')
                valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            # Make the trainer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            if patience is None:
                patience = num_epochs // 8
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.8, min_lr=1e-6)

            # Store the best loss
            best_loss = torch.inf

            # Loop over epochs
            log = []
            model.train()
            model.to(device)
            start_time = time.perf_counter()
            for epoch in range(num_epochs):
                # Iterate over all batches in the training set
                train_losses = defaultdict(list)
                for batch in train_loader:
                    batch.to(device)

                    optimizer.zero_grad()

                    # Compute the energy and forces
                    energy, force = eval_batch(model, batch)

                    # Get the forces in energy and forces
                    energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                    force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                    total_loss = energy_weight * energy_loss + (1 - energy_weight) * force_loss

                    # Iterate backwards
                    total_loss.backward()
                    optimizer.step()

                    # Add the losses to a log
                    with torch.no_grad():
                        train_losses['train_loss_force'].append(force_loss.item())
                        train_losses['train_loss_energy'].append(energy_loss.item())
                        train_losses['train_loss_total'].append(total_loss.item())

                # Compute the average loss for the batch
                train_losses = dict((k, np.mean(v)) for k, v in train_losses.items())

                # Get the validation loss
                valid_losses = defaultdict(list)
                for batch in valid_loader:
                    batch.to(device)
                    energy, force = eval_batch(model, batch)

                    # Get the loss of this batch
                    energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                    force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                    total_loss = energy_weight * energy_loss + (1 - energy_weight) * force_loss

                    with torch.no_grad():
                        valid_losses['valid_loss_force'].append(force_loss.item())
                        valid_losses['valid_loss_energy'].append(energy_loss.item())
                        valid_losses['valid_loss_total'].append(total_loss.item())

                valid_losses = dict((k, np.mean(v)) for k, v in valid_losses.items())

                # Reduce the learning rate
                scheduler.step(valid_losses['valid_loss_total'])

                # Save the best model if possible
                if valid_losses['valid_loss_total'] < best_loss:
                    best_loss = valid_losses['valid_loss_total']
                    torch.save(model, td / 'best_model')

                # Store the log line
                log.append({'epoch': epoch, 'time': time.perf_counter() - start_time, **train_losses, **valid_losses})

            # Load the best model back in
            best_model = torch.load(td / 'best_model', map_location='cpu')

            return TorchMessage(best_model), pd.DataFrame(log)
