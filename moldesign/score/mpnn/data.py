"""Utilities for creating a data-loader"""
from functools import partial
from typing import List, Tuple, Sequence, Optional

import tensorflow as tf
import networkx as nx
import numpy as np

from moldesign.utils.conversions import convert_string_to_dict


def _numpy_to_tf_feature(value):
    """Converts a Numpy array to a Tensoflow Feature
 the
    Determines the dtype and ensures the array is at least 1D

    Args:
        value (np.array): Value to convert
    Returns:
        (tf.train.Feature): Feature representation of this full value
    """

    # Make sure value is an array, then flatten it to a 1D vector
    value = np.atleast_1d(value).flatten()

    if value.dtype.kind == 'f':
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    elif value.dtype.kind in ['i', 'u']:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        # Just send the bytes (warning: untested!)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


def make_tfrecord(network):
    """Make and serialize a TFRecord for in NFP format

    Args:
        network (dict): Network description as a dictionary
    Returns:
        (bytes) Record as a serialized string
    """

    # Convert the data to TF features
    features = dict((k, _numpy_to_tf_feature(v)) for k, v in network.items())

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def parse_records(example_proto, target_name: str = 'pIC50',
                  target_shape: Sequence[int] = ()):
    """Parse data from the TFRecord

    Args:
        example_proto: Batch of serialized TF records
        target_name (str): Name of the output property
        target_shape
    Returns:
        Batch of parsed TF records
    """

    default_target = np.zeros(target_shape) * np.nan

    features = {
        target_name: tf.io.FixedLenFeature(target_shape, tf.float32, default_value=default_target),
        'n_atom': tf.io.FixedLenFeature([], tf.int64),
        'n_bond': tf.io.FixedLenFeature([], tf.int64),
        'connectivity': tf.io.VarLenFeature(tf.int64),
        'atom': tf.io.VarLenFeature(tf.int64),
        'bond': tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_example(example_proto, features)


def prepare_for_batching(dataset):
    """Make the variable length arrays into RaggedArrays.

    Allows them to be merged together in batches"""
    for c in ['atom', 'bond', 'connectivity']:
        expanded = tf.expand_dims(dataset[c].values, axis=0, name=f'expand_{c}')
        dataset[c] = tf.RaggedTensor.from_tensor(expanded).flat_values
    return dataset


def combine_graphs(batch):
    """Combine multiple graphs into a single network"""

    # Compute the mappings from bond index to graph index
    batch_size = tf.size(batch['n_atom'], name='batch_size')
    mol_id = tf.range(batch_size, name='mol_inds')
    batch['node_graph_indices'] = tf.repeat(mol_id, batch['n_atom'], axis=0)
    batch['bond_graph_indices'] = tf.repeat(mol_id, batch['n_bond'], axis=0)

    # Reshape the connectivity matrix to (None, 2)
    batch['connectivity'] = tf.reshape(batch['connectivity'], (-1, 2))

    # Compute offsets for the connectivity matrix
    offset_values = tf.cumsum(batch['n_atom'], exclusive=True)
    offsets = tf.repeat(offset_values, batch['n_bond'], name='offsets', axis=0)
    batch['connectivity'] += tf.expand_dims(offsets, 1)

    return batch


def make_training_tuple(batch, target_name='pIC50'):
    """Get the output tuple.

    Makes a tuple dataset with the inputs as the first element
    and the output energy as the second element
    """

    inputs = {}
    output = None
    for k, v in batch.items():
        if k != target_name:
            inputs[k] = v
        else:
            output = v
    return inputs, output


def make_data_loader(file_path, batch_size=32, shuffle_buffer=None,
                     n_threads=tf.data.experimental.AUTOTUNE, shard=None,
                     cache: bool = False, output_property: str = 'pIC50',
                     output_shape: Sequence[int] = (), random_seed: Optional[int] = None) -> tf.data.TFRecordDataset:
    """Make a data loader for tensorflow

    Args:
        file_path (str): Path to the training set
        batch_size (int): Number of graphs per training batch
        shuffle_buffer (int): Width of window to use when shuffling training entries
        n_threads (int): Number of threads over which to parallelize data loading
        cache (bool): Whether to load the whole dataset into memory
        shard ((int, int)): Parameters used to shared the dataset: (size, rank)
        output_property (str): Which property to use as the output
        output_shape ([int]): Shape of the output property
    Returns:
        (tf.data.TFRecordDataset) An infinite dataset generator
    """

    r = tf.data.TFRecordDataset(file_path)

    # Save the data in memory if needed
    if cache:
        r = r.cache()

    # Shuffle the entries
    if shuffle_buffer is not None:
        r = r.shuffle(shuffle_buffer, seed=random_seed)

    # Shard after shuffling (so that each rank will be able to make unique batches each time)
    if shard is not None:
        r = r.shard(*shard)

    # Add in the data preprocessing steps
    #  Note that the `batch` is the first operation
    parse = partial(parse_records, target_name=output_property, target_shape=output_shape)
    r = r.batch(batch_size).map(parse, n_threads).map(prepare_for_batching, n_threads)

    # Return full batches
    r = r.map(combine_graphs, n_threads)
    train_tuple = partial(make_training_tuple, target_name=output_property)
    return r.map(train_tuple)


def create_batches_from_objects(graphs: List[dict], batch_size: int = 32) -> List[dict]:
    """Create batches from a collection of graphs in dictionary format

    Args:
        graphs: List of graphs to make into batches
        batch_size: Number of graphs per batch
    Returns:
        Batches of graphs where the values are TF Tensors
    """

    # Combine graphs into chunks that will be made into batches
    chunks = []
    for start in range(0, len(graphs), batch_size):
        chunks.append(graphs[start:start+batch_size])

    # Combine graphs into chunks
    batches = []
    keys = chunks[0][0].keys()
    for chunk in chunks:
        batch_dict = {}
        for k in keys:
            batch_dict[k] = np.concatenate([np.atleast_1d(b[k]) for b in chunk], axis=0)
        batches.append(combine_graphs(batch_dict))

    return batches


def _merge_batch(mols: List[dict]) -> dict:
    """Merge a list of molecules into a single batch

    Args:
        mols: List of molecules in dictionary format
    Returns:
        Single batch of molecules
    """

    # Convert arrays to array

    # Stack the values from each array
    batch = dict(
        (k, np.concatenate([np.atleast_1d(m[k]) for m in mols], axis=0))
        for k in mols[0].keys()
    )

    # Compute the mappings from bond index to graph index
    batch_size = len(mols)
    mol_id = np.arange(batch_size, dtype=np.int)
    batch['node_graph_indices'] = np.repeat(mol_id, batch['n_atom'], axis=0)
    batch['bond_graph_indices'] = np.repeat(mol_id, batch['n_bond'], axis=0)

    # Compute offsets for the connectivity matrix
    offset_values = np.zeros(batch_size, dtype=np.int)
    np.cumsum(batch['n_atom'][:-1], out=offset_values[1:])
    offsets = np.repeat(offset_values, batch['n_bond'], axis=0)
    batch['connectivity'] += np.expand_dims(offsets, 1)

    return batch


class GraphLoader(tf.keras.utils.Sequence, tf.keras.callbacks.Callback):
    """Keras-compatible data loader for training a graph problem

    Styled after the NFP TFv1 data loader: https://github.com/NREL/nfp/blob/0.0.x/nfp/preprocessing/sequence.py"""

    def __init__(self, smiles: List[str], outputs: List[float], batch_size: int,
                 shuffle: bool = True, random_state: int = None):
        """

        Args:
            smiles: List of molecules
            outputs: List of molecular outputs
            batch_size: Number of batches to use to train model
            shuffle: Whether to shuffle after each epoch
            random_state: Random state for the shuffling
        """

        super(GraphLoader, self).__init__()

        # Convert the molecules to MPNN-ready formats
        mols = [convert_string_to_dict(s) for s in smiles]
        self.entries = np.array(list(zip(mols, outputs)))

        # Other data
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Give it a first shuffle, if needed
        self.rng = np.random.RandomState(random_state)
        if shuffle:
            self.rng.shuffle(self.entries)

    def on_epoch_end(self, epoch=None, log=None):
        if self.shuffle:
            # Shuffle the dataset
            self.rng.shuffle(self.entries)

    def __getitem__(self, item):
        # Get the desired chunk of entries
        start = item * self.batch_size
        chunk = self.entries[start:start + self.batch_size]

        # Get the molecules and outputs out
        mols, y = zip(*chunk)
        x = _merge_batch(mols)
        return x, np.array(y)[:, None]

    def __len__(self):
        # Get the number of batches
        train_size = len(self.entries)
        n_batches = train_size // self.batch_size

        # Add a partially-full batch at the end
        if train_size % self.batch_size != 0:
            n_batches += 1
        return n_batches
