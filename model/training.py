import logging
import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def train_and_evaluate(model, data_set, log_dir, hparams, params, run_name, restore_from=None, store_checkpoint=False):
    """Train the model and evaluate every epoch.

    Args:
        model: tf model to be trained
        data_set: (dict) containing train_ds, val_ds, test_ds tf datsets
        log_dir: (string) log directory path
        hparams: (dict) containing hyper-parameters to be traced out in logs
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs
        run_name: (string) Name of the training run
        restore_from: (string) directory or file containing weights to restore the graph
                Optional
        store_checkpoint: (bool) Flag to control checkpoints save.
                Optional
    """
    # Initialize tf.Saver instances to save weights during training
    file = log_dir + '/logs/fit/' + run_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=file, histogram_freq=0)
    hpboard_callback = hp.KerasCallback(file, hparams)
    print('FileName')
    print(file)

    checkpoint_path = log_dir + '/checkpoints'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    train_ds = data_set['train_ds']
    val_ds = data_set['val_ds']
    test_ds = data_set['test_ds']
    METRIC_ACCURACY = 'accuracy'

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    with tf.summary.create_file_writer(file).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        model.fit(train_ds, epochs=params.num_epochs, validation_data = val_ds, callbacks=[tensorboard_callback, hpboard_callback])
        oss, accuracy = model.evaluate(test_ds)
        #accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
