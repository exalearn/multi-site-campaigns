"""Sampling structures using molecular dynamics"""
import os
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
from typing import Optional

from ase.calculators.calculator import Calculator
from ase.md import VelocityVerlet
from ase.io import Trajectory
from ase import units
import ase

from .base import CalculatorBasedSampler

logger = logging.getLogger(__name__)


class MolecularDynamics(CalculatorBasedSampler):
    """Sample new structures using molecular dynamics"""

    def _run_sampling(self, atoms: ase.Atoms, steps: int, calc: Calculator, timestep: float = 5, log_interval: Optional[int] = None)\
            -> (ase.Atoms, list[ase.Atoms]):
        """Run molecular dynamics to produce a list of sampled structures

        Args:
            atoms: Initial structure
            steps: Number of MD steps
            calc: Calculator used to compute energy and forces
            timestep: Timestep size (units: fs)
            log_interval: Number of steps between storing structures. Defaults to 100 structures per run
        Returns:
            - List of structures sampled along the way
        """

        # Set the calculator
        atoms.calc = calc

        # Define the dynamics
        dyn = VelocityVerlet(atoms, timestep=timestep * units.fs)

        # Determine the number of steps
        log_interval = max(1, steps // 100) if log_interval is None else log_interval

        # Store the trajectory data to a temporary directory
        with TemporaryDirectory(dir=self.scratch_dir, prefix='fff') as tmp:

            # Move to the temporary directory so that no files written to disk overlap
            start_dir = os.getcwd()
            try:
                os.chdir(tmp)

                # Define the output path
                traj_path = Path(tmp) / "md.traj"
                logger.info(f'Writing trajectory to {traj_path}')
                props_to_write = ['energy', 'forces']
                with Trajectory(str(traj_path), mode='w', atoms=atoms, properties=props_to_write) as traj:
                    dyn.attach(traj, interval=log_interval)

                    # Run the dynamics
                    dyn.run(steps)

                # Read the trajectory back in and return the atoms
                return atoms, [x for x in Trajectory(str(traj_path), mode='r')][1:-1]
            finally:
                os.chdir(start_dir)
