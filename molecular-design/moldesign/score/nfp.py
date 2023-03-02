"""Interfaces to run `NFP <https://github.com/NREL/nfp>`_ models through Colmena workflows"""
from typing import List, Any, Optional, Tuple, Dict, Union
import pickle as pkl

import nfp
from tensorflow.python.keras import callbacks as cb
import tensorflow as tf
import numpy as np

from moldesign.utils.callbacks import LRLogger, EpochTimeLogger, TimeLimitCallback
from moldesign.utils.conversions import convert_string_to_dict


class ReduceAtoms(tf.keras.layers.Layer):
    """Reduce the atoms along a certain direction

    Args:
        reduction_op: Name of the operation used for reduction
    """

    def __init__(self, reduction_op: str = 'mean', **kwargs):
        super().__init__(**kwargs)
        self.reduction_op = reduction_op

    def get_config(self):
        config = super().get_config()
        config['reduction_op'] = self.reduction_op
        return config

    def call(self, inputs, mask=None):
        masked_tensor = tf.ragged.boolean_mask(inputs, mask)
        reduce_fn = getattr(tf.math, f'reduce_{self.reduction_op}')
        return reduce_fn(masked_tensor, axis=1)


# Define the custom layers for our class
custom_objects = nfp.custom_objects.copy()
custom_objects['ReduceAtoms'] = ReduceAtoms


def _to_nfp_dict(x: dict) -> dict:
    """Convert a moldesign-compatible dict to one usable by NFP

    Removes the ``n_atom`` and ``n_bond`` keys, and increments the bond type and atom type by
    1 because nfp uses 0 as a padding mask

    Args:
        x: Dictionary to be modified
    Returns:
        The input dictionary
    """

    x = x.copy()
    for k in ['atom', 'bond']:
        x[k] = np.add(x[k], 1)
    for k in ['n_atom', 'n_bond']:
        x.pop(k, None)
    return x


class NFPMessage:
    """Package for sending an MPNN model over connections that require pickling"""

    def __init__(self, model: tf.keras.Model):
        """
        Args:
            model: Model to be sent
        """

        self.config = model.to_json()
        # Makes a copy of the weights to ensure they are not memoryview objects
        self.weights = [np.array(v) for v in model.get_weights()]

        # Cached copy of the model
        self._model = model

    def __getstate__(self):
        """Get state except the model"""
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def get_model(self) -> tf.keras.Model:
        """Get a copy of the model

        Returns:
            The model specified by this message
        """
        if self._model is None:
            self._model = tf.keras.models.model_from_json(
                self.config,
                custom_objects=custom_objects
            )
            self._model.set_weights(self.weights)
        return self._model


def make_data_loader(mol_dicts: List[dict],
                     values: Optional[List[Any]] = None,
                     batch_size: int = 32,
                     repeat: bool = False,
                     shuffle_buffer: Optional[int] = None,
                     value_spec: tf.TensorSpec = tf.TensorSpec((), dtype=tf.float32),
                     max_size: Optional[int] = None,
                     drop_last_batch: bool = False) -> tf.data.Dataset:
    """Make a data loader for data compatible with NFP-style neural networks

    Args:
        mol_dicts: List of molecules parsed into the moldesign format
        values: List of output values, if included in the output
        value_spec: Tensorflow specification for the output
        batch_size: Number of molecules per batch
        repeat: Whether to create an infinitely-repeating iterator
        shuffle_buffer: Size of a shuffle buffer. Use ``None`` to leave data unshuffled
        max_size: Maximum number of atoms per molecule
        drop_last_batch: Whether to keep the last batch in the dataset. Set to ``True`` if, for example, you need every batch to be the same size
    Returns:
        Data loader that generates molecules in the desired shapes
    """

    # Convert to NFP format
    mol_dicts = [_to_nfp_dict(x) for x in mol_dicts]

    # Make the initial data loader
    record_sig = {
        "atom": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "bond": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "connectivity": tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
    }
    if values is None:
        def generator():
            yield from mol_dicts
    else:
        def generator():
            yield from zip(mol_dicts, values)

        record_sig = (record_sig, value_spec)

    loader = tf.data.Dataset.from_generator(generator=generator, output_signature=record_sig).cache()  # TODO (wardlt): Make caching optional?

    # Repeat the molecule list before shuffling
    if repeat:
        loader = loader.repeat()

    # Shuffle, if desired
    if shuffle_buffer is not None:
        loader = loader.shuffle(shuffle_buffer)

    # Make the batches
    #  Patches the data to make them all the same size, adding 0's to signify padded values
    if max_size is None:
        loader = loader.padded_batch(batch_size=batch_size, drop_remainder=drop_last_batch)
    else:
        max_bonds = 4 * max_size  # If all atoms are carbons, they each have 4 bonds at maximum
        padded_records = {
            "atom": tf.TensorShape((max_size,)),
            "bond": tf.TensorShape((max_bonds,)),
            "connectivity": tf.TensorShape((max_bonds, 2))
        }
        if values is not None:
            padded_records = (padded_records, value_spec.shape)
        loader = loader.padded_batch(batch_size=batch_size, padded_shapes=padded_records, drop_remainder=drop_last_batch)

    return loader


def evaluate_mpnn(
        model_msg: NFPMessage,
        mol_dicts: List[dict],
        batch_size: int = 128,
        max_size: Optional[int] = None
) -> np.ndarray:
    """Run inference on a list of molecules

    Args:
        model_msg: MPNN to evaluate. Accepts a pickled message
        mol_dicts: List of molecules as MPNN-ready dictionary objections
        batch_size: Number of molecules per batch
        max_size: Maximum size of the molecules
    Returns:
        Predicted value for each molecule
    """
    assert len(mol_dicts) > 0, "You must provide at least one molecule to inference function"

    # Unpack the messages
    model = model_msg.get_model()

    # Determine the maximum molecule size
    if max_size is None:
        max_size = max(x['n_atom'] for x in mol_dicts)

    # Make a dataloader
    loader = make_data_loader(
        mol_dicts,
        batch_size=batch_size,
        repeat=False,
        max_size=max_size,
    )

    # Run
    return np.squeeze(model.predict(loader))


def device_strategy(device: str = 'gpu', num_devices: Union[int, list, tuple] = 1) -> Tuple[tf.distribute.Strategy, int]:
    """Create the strategy for parallelizing training across multiple devices

    Args:
        device: Type of device
        num_devices: Target number of devices to use (needed for IPUs)
    Returns:
        - Parallelization strategy
        - Number of devices selected
    """
    if device == 'ipu':
        #  Configure the IPU system and define the strategy
        cfg = ipu.config.IPUConfig()
        if type(num_devices) == int:
            print("Automatically selecting the available IPUs")
            cfg.auto_select_ipus = num_devices
        else:
            cfg.select_ipus = list(num_devices)
        cfg.configure_ipu_system()

        strategy = ipu.ipu_strategy.IPUStrategy(enable_dataset_iterators=True)
    elif device == 'gpu':
        # Distribute over all available GPUs
        strategy = tf.distribute.MirroredStrategy()

        # Print the GPU list
        device_details = [
            tf.config.experimental.get_device_details(x)
            for x in tf.config.get_visible_devices('GPU')
        ]
        return strategy, len(device_details)
    else:
        raise ValueError(f'Device type "{device}" not supported yet')

    return strategy


def retrain_mpnn(model_config: dict,
                 database: Dict[str, float],
                 num_epochs: int,
                 batch_size: int = 32,
                 validation_split: float = 0.1,
                 random_state: int = 1,
                 learning_rate: float = 1e-3,
                 device_type: str = 'gpu',
                 steps_per_exec: int = 1,
                 patience: int = None,
                 timeout: float = None,
                 verbose: bool = False
                 ) \
        -> Union[Tuple[List, dict], Tuple[List, dict, List[float]]]:
    """Train a model from initialized weights

    Args:
        model_config: Serialized version of the model
        database: Training dataset of molecule mapped to a property
        num_epochs: Maximum number of epochs to run
        batch_size: Number of molecules per training batch
        validation_split: Fraction of molecules used for the training/validation split
        random_state: Seed to the random number generator used for splitting data
        learning_rate: Learning rate for the Adam optimizer
        device_type: Type of device used for training
        steps_per_exec: Number of training steps to run per execution on acceleration
        patience: Number of epochs without improvement before terminating training.
        timeout: Maximum training time in seconds
        verbose: Whether to print training information to screen
    Returns:
        - model: Updated weights
        - history: Training history
        - test_pred: Prediction on test set, if provided
    """
    # Update some defaults
    if patience is None:
        patience = max(num_epochs // 8, 1)

    # Make the data loaders
    # Separate the database into molecules and properties
    smiles, y = zip(*database.items())
    mol_dicts = np.array([convert_string_to_dict(s) for s in smiles])
    max_size = max(len(x['atom']) for x in mol_dicts)
    y = np.array(y)

    # Make the training and validation splits
    rng = np.random.RandomState(random_state)
    train_split = rng.rand(len(smiles)) > validation_split
    train_X = mol_dicts[train_split]
    train_y = y[train_split]
    valid_X = mol_dicts[~train_split]
    valid_y = y[~train_split]

    # Make the loaders
    steps_per_epoch = len(train_X) // batch_size
    train_loader = make_data_loader(train_X, train_y, repeat=True, batch_size=batch_size, max_size=max_size, drop_last_batch=True, shuffle_buffer=32768)
    valid_steps = len(valid_X) // batch_size
    valid_loader = make_data_loader(valid_X, valid_y, batch_size=batch_size, max_size=max_size, drop_last_batch=True)

    # Make the strategy for parallelization the training across multiple devices
    strategy, num_devices = device_strategy(device_type, 1)

    with strategy.scope():
        # Make a copy of the model
        model: tf.keras.models.Model = tf.keras.models.Model.from_config(model_config, custom_objects=custom_objects)

        # Define initial guesses for the "scaling" later
        try:
            scale_layer = model.get_layer('scale')
            outputs = np.array(list(database.values()))
            scale_layer.set_weights([outputs.std()[None, None], outputs.mean()[None]])
        except ValueError:
            pass

        # Asynchronous callback option on
        if device_type == "ipu" and steps_per_exec > 1:
            model.set_asynchronous_callbacks(asynchronous=True)

        # Configure the LR schedule
        init_learn_rate = learning_rate
        final_learn_rate = init_learn_rate * 1e-3
        decay_rate = (final_learn_rate / init_learn_rate) ** (1. / (num_epochs - 1))

        def lr_schedule(epoch, lr):
            return lr * decay_rate

        # Compile the model then train
        model.compile(
            tf.optimizers.Adam(init_learn_rate),
            'mean_squared_error',
            metrics=['mean_absolute_error'],
            steps_per_execution=steps_per_exec,
        )

        # Make the callbacks
        early_stopping = cb.EarlyStopping(patience=patience, restore_best_weights=True)
        callbacks = [
            LRLogger(),
            EpochTimeLogger(),
            cb.LearningRateScheduler(lr_schedule),
            early_stopping,
            cb.TerminateOnNaN(),
        ]
        if timeout is not None:
            callbacks.append(TimeLimitCallback(timeout))

        history = model.fit(
            train_loader,
            epochs=num_epochs,
            shuffle=False,
            verbose=verbose,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_loader,
            validation_steps=valid_steps,
            validation_freq=1,
        )

    # If a timeout is used, make sure we are using the best weights
    #  The training may have exited without storing the best weights
    if timeout is not None:
        model.set_weights(early_stopping.best_weights)

    # Convert weights to numpy arrays (avoids mmap issues)
    weights = []
    for v in model.get_weights():
        v = np.array(v)
        if np.isnan(v).any():
            raise ValueError('Found some NaN weights.')
        weights.append(v)

    # Once we are finished training call "clear_session"
    tf.keras.backend.clear_session()
    return weights, history.history
