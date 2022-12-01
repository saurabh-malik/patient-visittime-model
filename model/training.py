import logging
import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def train_and_evaluate(model, data_set, log_dir, hparams, params, run_name, restore_from=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training


    file = log_dir + '/' + run_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=file, histogram_freq=0)
    hpboard_callback = hp.KerasCallback(file, hparams)

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
