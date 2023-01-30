"""Base class for simulation methods"""
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import ase
from ase.calculators.calculator import Calculator

from fff.learning.gc.ase import SchnetCalculator


class BaseSampler:
    """Defines the settings and method to run a simulation that generates new atomic structures"""

    def __init__(self, scratch_dir: Optional[Path] = None):
        """
        Args:
            scratch_dir: Path in which to store temporary files
        """

        self.scratch_dir = scratch_dir

    @abstractmethod
    def run_sampling(self, atoms: ase.Atoms, steps: int, **kwargs) -> (ase.Atoms, list[ase.Atoms]):
        """Run a sampling method

        Args:
            atoms: Starting structure
            steps: Number of iterations over which to run the sampler.
                Larger numbers should produce more structures or a greater variety
        Returns:
            - Structure used to audit the sampling performance
            - List of new structures to consider
        """
        ...


class CalculatorBasedSampler(BaseSampler):
    """A sampler class which uses an ase :class:`~ase.calculators.calculator.Calculator` to generate energies"""

    def run_sampling(self, atoms: ase.Atoms, steps: int, calc: Calculator = None,
                     device: Optional[str] = None, **kwargs) -> (ase.Atoms, list[ase.Atoms]):
        """Run a sampling method

        Args:
            atoms: Starting structure
            steps: Number of iterations over which to run the sampler.
                Larger numbers should produce more structures or a greater variety
            calc: Calculator used to generate energies
            device: Device on which to run the calculator (e.g., a GPU)
        Returns:
            - Structure used to audit the sampling performance
            - List of new structures to consider
        """
        assert calc is not None, 'You must specify a calculator'

        # Unpack the calculator depending on its class
        if isinstance(calc, SchnetCalculator):
            calc.to(device)

        return self._run_sampling(atoms, steps, calc, **kwargs)

    @abstractmethod
    def _run_sampling(self, atoms: ase.Atoms, steps: int, calc: Calculator, **kwargs) -> (ase.Atoms, list[ase.Atoms]):
        """Private method that must be implemented"""
        ...
