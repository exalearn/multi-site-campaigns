from pathlib import Path

from qcelemental.models import OptimizationResult, AtomicResult

from moldesign.store.models import MoleculeData, OxidationState
from moldesign.store.recipes import get_recipe_by_name


_my_path = Path(__file__).parent


def test_get_required():
    md = MoleculeData.from_identifier("O")

    # Show that we need at least a relaxation for to complete xtb-vacuum-vertical
    recipe = get_recipe_by_name('xtb-vacuum-vertical')

    required = recipe.get_required_calculations(md, OxidationState.OXIDIZED)
    assert len(required) == 1
    assert required[0] == (
        'xtb',
        required[0][1],  # An XYZ file
        0,
        None,
        True
    )
    assert required[0][1].startswith('3')

    # After performing that relaxation, we need the computation of the energy in the charged geometry
    xtb_geom = OptimizationResult.parse_file(_my_path.joinpath('records/xtb-neutral.json'))
    md.add_geometry(xtb_geom)

    required = recipe.get_required_calculations(md, OxidationState.OXIDIZED)
    assert len(required) == 1
    assert required[0] == (
        'xtb',
        xtb_geom.final_molecule.to_string('xyz'),
        1,
        None,
        False
    )

    # After performing that computation, we should not have anything left to do for the xtb-vacuum-vertical
    xtb_energy = AtomicResult.parse_file(_my_path.joinpath('records/xtb-neutral_xtb-oxidized-energy.json'))
    md.add_single_point(xtb_energy)

    required = recipe.get_required_calculations(md, OxidationState.OXIDIZED)
    assert len(required) == 0

    # If we need to do xtb-vacuum, our next step is to relax the geometry
    recipe = get_recipe_by_name('xtb-vacuum')

    required = recipe.get_required_calculations(md, OxidationState.OXIDIZED)
    assert len(required) == 1
    assert required[0] == (
        'xtb',
        md.data['xtb'][OxidationState.NEUTRAL].xyz,
        1,
        None,
        True   # We require a relaxation
    )

    # Once we do that oxidized geometry, nothing else is required
    xtb_geom_ox = OptimizationResult.parse_file(_my_path.joinpath('records/xtb-oxidized.json'))
    md.add_geometry(xtb_geom_ox)

    required = recipe.get_required_calculations(md, OxidationState.OXIDIZED)
    assert len(required) == 0

    # We need to do two solvation energy computations to finish the XTB-acn recipe
    recipe = get_recipe_by_name('xtb-acn')

    required = recipe.get_required_calculations(md, OxidationState.OXIDIZED)
    assert len(required) == 2
    assert set(x[0] for x in required) == {'xtb'}
    assert set(x[1] for x in required) == {md.data['xtb'][OxidationState.NEUTRAL].xyz,
                                           md.data['xtb'][OxidationState.OXIDIZED].xyz}
    assert set(x[2] for x in required) == {0, 1}
    assert set(x[3] for x in required) == {'acetonitrile'}
    assert set(x[4] for x in required) == {False}

    # See that we need two relaxations for going up to smb-vacuum-no-zpe
    recipe = get_recipe_by_name('smb-vacuum-no-zpe')

    required = recipe.get_required_calculations(md, OxidationState.OXIDIZED,
                                                previous_level=get_recipe_by_name('xtb-vacuum'))
    assert len(required) == 2
    assert set(x[0] for x in required) == {'small_basis'}
    assert set(x[1] for x in required) == {md.data['xtb'][OxidationState.NEUTRAL].xyz,
                                           md.data['xtb'][OxidationState.OXIDIZED].xyz}
    assert set(x[2] for x in required) == {0, 1}
    assert set(x[3] for x in required) == {None}
    assert set(x[4] for x in required) == {True}
