"""Utilities that build atop ASE"""
from typing import Optional
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import logging


import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, CalculationFailed
from ase.db import connect
from parsl.app.python import PythonApp

logger = logging.getLogger(__name__)


def _run_calculator(calc: Calculator, atoms: Atoms, *args) -> Calculator:
    """Run a calculator and return the updated value"""

    calc.calculate(atoms, *args)
    return calc


def get_atoms_hash(atoms: Atoms) -> str:
    """Get a string hash for a structure

    Bases a hash on the same things ASE uses to test for equality: atomic positions and types, cell parameters, and PBC.

    Args:
        atoms: Atoms to be hashed
    Returns:
        Hashed output
    """

    sha = sha256()
    sha.update(np.array(atoms.positions).tobytes('C'))
    sha.update(np.array(atoms.numbers).tobytes('C'))
    sha.update(atoms.cell.array.tobytes('C'))
    sha.update(atoms.pbc.tobytes('C'))
    return sha.hexdigest()[:32]


class CachedCalculator(Calculator):
    """Calculator that stores all results to an ASE DB once complete

    Will also look up results from the database and return any exact matches
    """

    def __init__(self, sub_calc: Calculator, db_path: Path):
        """
        Args:
             sub_calc: Calculator actually used to perform computations
             db_path: Path to the database file
        """
        super().__init__()
        self.implemented_properties = sub_calc.implemented_properties
        self.sub_calc = sub_calc
        self.db_path = db_path
        self.last_lookup = False  # Whether the last computation used the cache

    def get_db_size(self) -> int:
        """Get the size of the cache"""
        with connect(self.db_path, type='db') as db:
            return db.count()

    def calculate(self, atoms: Atoms = None, properties=None, system_changes=all_changes):
        # Get the target properties
        if properties is None:
            properties = self.implemented_properties

        # Get a hash for this atoms
        my_hash = get_atoms_hash(atoms)

        # Connect to the database
        with connect(self.db_path, type='db') as db:
            # Check to see if the result is there by looping over all structures with the same hash
            match: Optional[Atoms] = None
            for row in db.select(atomshash=my_hash):
                other_atoms = row.toatoms()
                if other_atoms == atoms:
                    match = other_atoms
                    break

            # If we match, store the properties of this cached computation
            if match:
                self.results.update(match.get_properties(properties))
                self.last_lookup = True
                logger.info(f'Found a cached calculation. Using results from {self.db_path}')
                return
            self.last_lookup = False

            # If not, call the underlying calculator
            start_time = perf_counter()
            self.sub_calc.calculate(atoms, properties, system_changes)
            run_time = perf_counter() - start_time

            # Store the result
            to_store = atoms.copy()
            to_store.calc = self.sub_calc
            db.write(to_store, atomshash=my_hash, run_time=run_time)
            logger.info(f'Saved new calculation to {self.db_path}. New size: {db.count()}')

            # Store the results in our class
            self.results.update(self.sub_calc.results)


class AsyncCalculator(Calculator):
    """Calculator that performs a computation on an external resource"""

    def __init__(self, sub_calc: Calculator):
        """

        Args:
            sub_calc: Calculator to be run asynchronously
        """
        super().__init__()
        self.func = PythonApp(_run_calculator)
        self.sub_calc = sub_calc
        self.implemented_properties = sub_calc.implemented_properties

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        # Get the target properties
        if properties is None:
            properties = self.implemented_properties

        # Run it using Parsl
        future = self.func(self.sub_calc, atoms, properties, system_changes)
        try:
            self.sub_calc = future.result()
        except CalculationFailed as exc:
            #  If the error was due to a process being killed, resubmit
            if 'code 134' in str(exc):
                future = self.func(self.sub_calc, atoms, properties, system_changes)
            self.sub_calc = future.result()

        # Store the results in our wrapper class
        self.results.update(self.sub_calc.results)
