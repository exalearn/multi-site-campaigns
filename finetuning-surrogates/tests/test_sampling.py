from pathlib import Path

import ase
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fff.sampling.md import MolecularDynamics


def test_md(atoms):
    calc = LennardJones()
    MaxwellBoltzmannDistribution(atoms, temperature_K=60)
    md = MolecularDynamics()
    atoms, traj = md.run_sampling(atoms, 1000, calc, timestep=1, log_interval=100)
    assert len(traj) == 9

    # Make sure it has both the energy and the forces
    assert isinstance(traj[0].calc, SinglePointCalculator)  # SPC is used to store results
    assert traj[0].get_forces().shape == (3, 3)
    assert traj[0].get_total_energy()
    assert traj[0].get_total_energy() == traj[-1].get_total_energy()
