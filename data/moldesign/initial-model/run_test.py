"""Train the model on a pre-defined training set and options provided by a user"""
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union
from time import perf_counter
import hashlib
import json

from sklearn.model_selection import train_test_split

from moldesign.store.models import MoleculeData
from moldesign.utils.conversions import convert_string_to_dict

try:
    from tensorflow.python import ipu
except ImportError:
    print('Cannot find the `ipu` library for Tensorflow')
    ipu = None

from moldesign.score.nfp import retrain_mpnn, evaluate_mpnn, NFPMessage, ReduceAtoms
from tensorflow.keras import layers
import tensorflow as tf
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import numpy as np
import nfp


def build_fn(
        atom_features: int = 64,
        message_steps: int = 8,
        output_layers: List[int] = (512, 256, 128),
        reduce_op: str = 'mean',
        atomwise: bool = True,
) -> tf.keras.models.Model:
    """Construct a Keras model using the settings provided by a user

    Args:
        atom_features: Number of features used per atom and bond
        message_steps: Number of message passing steps
        output_layers: Number of neurons in the readout layers
        reduce_op: Operation used to reduce from atom-level to molecule-level vectors
    Returns:

    """
    atom = layers.Input(shape=[None], dtype=tf.int32, name='atom')
    bond = layers.Input(shape=[None], dtype=tf.int32, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity')

    # Convert from a single integer defining the atom state to a vector
    # of weights associated with that class
    atom_state = layers.Embedding(64, atom_features, name='atom_embedding', mask_zero=True)(atom)

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
    if not atomwise:
        output = ReduceAtoms(reduce_op)(output)
    for shape in output_layers:
        output = layers.Dense(shape, activation='relu')(output)
    output = layers.Dense(1)(output)
    if atomwise:
        output = ReduceAtoms(reduce_op)(output)
    output = layers.Dense(1, activation='linear', name='scale')(output)

    # Construct the tf.keras model
    return tf.keras.Model([atom, bond, connectivity], [output])


def parse_args():
    # Define the command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--atom-features', help='Number of atomic features', type=int, default=32)
    arg_parser.add_argument('--num-messages', help='Number of message-passing layers', type=int, default=8)
    arg_parser.add_argument('--output-layers', help='Number of hidden units of the output layers', type=int,
                            default=(512, 256, 128), nargs='*')
    arg_parser.add_argument('--batch-size', help='Number of molecules per batch', type=int, default=16)
    arg_parser.add_argument('--num-epochs', help='Number of epochs to run', type=int, default=64)
    arg_parser.add_argument('--system', choices=['gpu', 'ipu'], help='Which system to use for training', default='gpu')
    arg_parser.add_argument('--lr-start', default=1e-3, help='Learning rate at start of training', type=float)
    arg_parser.add_argument('--reduce-op', default='sum', help='Operation used to reduce from atom to molecule features')
    arg_parser.add_argument('--atomwise', action='store_true', help='Reduce of atomic contributions')

    # Parse the arguments
    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    run_params = args.__dict__
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]

    # Determine the output directory
    test_dir = Path('networks') / f'{args.system}_b{args.batch_size}_n{args.num_epochs}_R{args.reduce_op}_{params_hash}'
    test_dir.mkdir(parents=True, exist_ok=True)
    with open(test_dir / 'config.json', 'w') as fp:
        json.dump(run_params, fp)

    # Load in the data
    with open('../training-data.json') as fp:
        all_X = []
        all_y = []
        for line in fp:
            record = MoleculeData.parse_raw(line)
            all_X.append(record.identifier['smiles'])
            all_y.append(record.oxidation_potential['xtb-vacuum'])

    # Split into train and testing
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, train_size=0.9, random_state=1)

    # Get the maximum size of all molecules
    max_size = max(convert_string_to_dict(x)['n_atom'] for x in all_X)

    # Make the model
    model = build_fn(atom_features=args.atom_features, message_steps=args.num_messages,
                     output_layers=args.output_layers, reduce_op=args.reduce_op,
                     atomwise=args.atomwise)

    # Train it
    start_time = perf_counter()
    weights, logs = retrain_mpnn(
        model.get_config(),
        dict(zip(train_X, train_y)),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr_start,
        verbose=True
    )
    run_time = perf_counter() - start_time

    # Update the weights and run testing
    model.set_weights(weights)

    # Save the model
    model.save(test_dir / 'model.h5')

    # Save the training log
    pd.DataFrame(logs).to_csv(test_dir / 'train_log.csv', index=False)

    # Run the test
    test_pred = evaluate_mpnn(
        model_msg=NFPMessage(model),
        mol_dicts=[convert_string_to_dict(x) for x in test_X]
    )

    pd.DataFrame({'true': test_y, 'pred': test_pred}).to_csv(test_dir / 'test_results.csv', index=False)

    with open(test_dir / 'test_summary.json', 'w') as fp:
        json.dump({
            'runtime': run_time,
            'r2_score': float(np.corrcoef(test_y, test_pred)[1, 0] ** 2),  # float() converts from np.float32
            'spearmanr': float(spearmanr(test_y, test_pred)[0]),
            'kendall_tau': float(kendalltau(test_y, test_pred)[0]),
            'mae': float(np.mean(np.abs(test_y - test_pred))),
            'rmse': float(np.sqrt(np.mean(np.square(test_y - test_pred))))
        }, fp, indent=2)
