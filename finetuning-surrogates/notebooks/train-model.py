"""Train a model on the full dataset produced from the run"""
import json
import shutil
from random import shuffle
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
from ase import io
from ase import Atoms
import torch

from fff.learning.gc.functions import GCSchNetForcefield
from fff.learning.gc.models import SchNet

if __name__ == "__main__":

    # Get the path to the run
    parser = ArgumentParser()
    parser.add_argument('run_dir', help='Path to the output files')
    parser.add_argument('--num-epochs', default=128, type=int, help='Number of epochs during training')
    parser.add_argument('--overwrite', action='store_true', help='Delete old run and restart')
    parser.add_argument('--max-force', default=None, type=float, help='Maximum allowed force value')
    parser.add_argument('--train-atomref', action='store_true', help='Unfreeze the atomref layer')
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='Learning rate for the model')
    parser.add_argument('--patience', default=None, type=int, help='Number of epochs before lowering the learning rate')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    print(f'Processing results in {run_dir}')

    # Load in the training data
    db_path = run_dir / 'train.db'
    atoms = io.read(db_path, ':')
    shuffle(atoms)
    assert isinstance(atoms[0], Atoms)
    print(f'Read {len(atoms)} training entries')

    # Trim off high-force entries
    if args.max_force is not None:
        atoms = [a for a in atoms if np.linalg.norm(a.get_forces(), axis=-1).max() < args.max_force]
        print(f'Trimmed to {len(atoms)} entries with forces below {args.max_force:.1f}')

    # Load in the run parameters
    with (run_dir / 'runparams.json').open() as fp:
        run_params = json.load(fp)

    # Create the retraining directory
    retrain_dir = run_dir / 'final-model'
    if retrain_dir.exists():
        if not args.overwrite:
            print(f'Output file found: {retrain_dir}. Call with "--overwrite" to repeat retraining')
            exit()
        else:
            print(f'Deleting old run at: {retrain_dir}')
            shutil.rmtree(retrain_dir)

    # Load in the starting model
    starting_model_path = Path(run_dir) / 'starting_model.pth'
    print(f'Using model from {starting_model_path.resolve()}')
    model: SchNet = torch.load(starting_model_path, map_location='cpu')
    model.atom_ref.weight.requires_grad = args.train_atomref

    # Retrain the model
    schnet = GCSchNetForcefield()
    n_train = int(len(atoms) * 0.9)
    print(f'Training using {n_train} points')
    model, train_log = schnet.train(
        starting_model_path,
        atoms[:n_train],
        atoms[n_train:],
        learning_rate=args.learning_rate,
        huber_deltas=(0.1, 1),
        num_epochs=args.num_epochs,
        patience=args.patience,
        device='cuda',
    )

    # Get the predictions on the validation set
    eng_pred, _ = schnet.evaluate(model, atoms[n_train:], device='cuda')
    eng_true = [a.get_potential_energy() for a in atoms[n_train:]]
    n_atoms = [len(a) for a in atoms[n_train:]]

    # Save the results
    retrain_dir.mkdir()
    torch.save(model.get_model('cpu'), retrain_dir / 'model')
    train_log.to_csv(retrain_dir / 'train_log.csv', index=False)
    pd.DataFrame({'n_atoms': n_atoms, 'eng_pred': eng_pred, 'eng_true': eng_true}).to_csv(retrain_dir / 'validation_set.csv', index=False)
    print(f'Saved results to {retrain_dir}')

    # Save run parameter
    with (retrain_dir / 'params.json').open('w') as fp:
        json.dump(args.__dict__, fp)
