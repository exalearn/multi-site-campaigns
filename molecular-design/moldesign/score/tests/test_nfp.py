from typing import Tuple, List

from tensorflow.keras import layers
from pytest import fixture
import tensorflow as tf
import numpy as np
import nfp

from moldesign.score.nfp import make_data_loader, evaluate_mpnn, NFPMessage, retrain_mpnn, ReduceAtoms
from moldesign.utils.conversions import convert_string_to_dict


@fixture()
def example_data() -> Tuple[List[str], List[dict], List[float]]:
    smiles = ['C', 'CC', 'CCC', 'CO', 'CCO']
    mol_dicts = [convert_string_to_dict(x) for x in smiles]
    return smiles, mol_dicts, [1., 2., 3., 4., 5.]


@fixture()
def example_model(atom_features: int = 64,
                  message_steps: int = 8,
                  output_layers: List[int] = (512, 256, 128)) -> tf.keras.Model:
    """Construct a Keras model using the settings provided by a user

    Args:
        atom_features: Number of features used per atom and bond
        message_steps: Number of message passing steps
        output_layers: Number of neurons in the readout layers

    Returns:
        Model
    """
    atom = layers.Input(shape=[None], dtype=tf.int32, name='atom')
    bond = layers.Input(shape=[None], dtype=tf.int32, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity')

    # Convert from a single integer defining the atom state to a vector
    # of weights associated with that class
    atom_state = layers.Embedding(36, atom_features, name='atom_embedding', mask_zero=True)(atom)

    # Ditto with the bond state
    bond_state = layers.Embedding(5, atom_features, name='bond_embedding', mask_zero=True)(bond)

    # Here we use our first nfp layer. This is an attention layer that looks at
    # the atom and bond states and reduces them to a single, graph-level vector.
    # mum_heads * units has to be the same dimension as the atom / bond dimension
    global_state = nfp.GlobalUpdate(units=4, num_heads=1, name='problem')([atom_state, bond_state, connectivity])

    for _ in range(message_steps):  # Do the message passing
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(units=4, num_heads=1)(
            [atom_state, bond_state, connectivity, global_state]
        )
        global_state = layers.Add()([global_state, new_global_state])

    # Pass the global state through an output
    output = atom_state
    for shape in output_layers:
        output = layers.Dense(shape, activation='relu')(output)
    output = layers.Dense(1)(output)
    output = layers.Dense(1, activation='linear', name='scale')(output)
    output = ReduceAtoms()(output)

    # Construct the tf.keras model
    return tf.keras.Model([atom, bond, connectivity], [output])


def test_loader(example_data):
    # Test a basic loader
    _, mol_dicts, y = example_data
    loader = make_data_loader(mol_dicts, batch_size=2)
    batch = next(loader.take(1).as_numpy_iterator())

    assert np.equal(batch['atom'], np.array([[6, 1, 1, 1, 1, 0, 0, 0],  # CH4
                                             [6, 6, 1, 1, 1, 1, 1, 1]])).all()  # C2H6

    # Test one where we give some input values
    loader = make_data_loader(mol_dicts, values=y, batch_size=2)
    batch = next(loader.take(1).as_numpy_iterator())
    assert np.equal(batch[0]['atom'], np.array([[6, 1, 1, 1, 1, 0, 0, 0],  # CH4
                                                [6, 6, 1, 1, 1, 1, 1, 1]])).all()  # C2H6
    assert np.equal(batch[1], np.array([1., 2.])).all()

    # Test where I fix the number of atoms
    loader = make_data_loader(mol_dicts, values=y, batch_size=2, max_size=20)
    batch = next(loader.take(1).as_numpy_iterator())
    assert batch[0]['bond'].shape == (2, 80)

    # Test with shuffling
    loader = make_data_loader(mol_dicts, batch_size=2, max_size=20, shuffle_buffer=4)
    batch = next(loader.take(1).as_numpy_iterator())
    assert batch['bond'].shape == (2, 80)

    # Test with dropping the last batch
    #  Should result in a spec with a fixed size in every dimension
    loader = make_data_loader(mol_dicts, batch_size=2, max_size=20, shuffle_buffer=4, drop_last_batch=True)
    for key, spec in loader.element_spec.items():
        assert all(x is not None for x in spec.shape), key


def test_evaluate(example_data, example_model):
    # Convert the input data
    _, mol_dicts, x = example_data
    model_msg = NFPMessage(example_model)
    model_msg._model = None

    # Make sure we get results with the correct dtype
    y = evaluate_mpnn(model_msg, mol_dicts)
    assert y.dtype == np.float32
    assert y.shape == (len(x),)
    assert not np.isnan(y).any()

    # Make sure it is invariant to input size
    y_new = evaluate_mpnn(model_msg, mol_dicts, max_size=128)
    assert np.isclose(y, y_new).all()


def test_training(example_data, example_model):
    # Upack the inputs
    smiles, mol_dicts, x = example_data
    config = example_model.get_config()

    # Call the training function
    weights, logs = retrain_mpnn(
        config,
        database=dict(zip(smiles, x)),
        validation_split=0.5,
        num_epochs=2,
        batch_size=1,
    )
    example_model.set_weights(weights)
    assert 'lr' in logs
