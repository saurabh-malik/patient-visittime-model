"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
import numpy as np
import pandas as pd
import logging

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.utils import df_to_dataset
from model.model_fn import model_fn
from model.encoder import get_category_encoding_layer
from model.encoder import get_normalization_layer
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
from model.training import train_and_evaluate
from model.feature_generator import encode_feature
from model.data_processing import load_data


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    tf.random.set_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(
        os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    ## ToDo Move Data load and split into seprate module . Done
    #Dataload
    dataframe = load_data(args.data_dir, "/visitdataclassification-4300.csv")

    #Create Target Variable as One hot encode
    #onehotecode = pd.get_dummies(dataframe['TotalTimeInWindow-15'], prefix='timewindow', sparse=True)
    onehotecode = pd.get_dummies(dataframe['TotalTimeInWindow-15'], prefix='timewindow', sparse=True)
    y = onehotecode.values
    output_shape = y.shape[1]
    print(onehotecode)
    print('Values')
    print(y)

    print('output_shape: ', output_shape)

    #Drop Unused feature
    dataframe = dataframe.drop(columns=['TotalTimeInMin', 'VisitEnvelopeId', 'IsDropoffAppointment', 'TotalTimeInWindow-30', 'TotalTimeInWindow-15'])

    #Split Data
    
    train_x, hold_x, train_y, hold_y = train_test_split(dataframe, y, test_size=0.40)
    val_x, test_x, val_y, test_y = train_test_split(hold_x, hold_y, test_size=0.50)

    #train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])



    batch_size = params.batch_size
    ds = df_to_dataset(dataframe, y, batch_size=batch_size)

    ## ToDo Move Feature creation into seprate module. Done
    #Features
    numerical_features = ['AgeInMonths', 'Weight', 'NumberOfVisitsInProgress']
    categorical_features = [ 'AnimalClass', 'Sex', 'Day','AnimalBreed', 'AppointmentTypeName','AppointmentReasons']
    
    #Encoded features.
    all_inputs, encoded_features = encode_feature(numerical_features, categorical_features, ds)

    train_inputs = {'all_inputs': all_inputs, 'encoded_features': encoded_features}

    train_ds = df_to_dataset(train_x, train_y, batch_size=batch_size)
    val_ds = df_to_dataset(val_x, val_y, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test_x, test_y, shuffle=False, batch_size=batch_size)

    data_set = {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
    }

    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32,64,95,128,180,256]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.3, 0.8))
    HP_LEARNINGRATE = hp.HParam('learning_rate', hp.Discrete([1e-4, 1e-3, 1e-2]))
    METRIC_ACCURACY = 'accuracy'

    log_dir = args.model_dir + '/logs/fit'
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_LEARNINGRATE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    session_num = 0

    for num_units in HP_NUM_UNITS.domain.values:
      for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for learning_rate in HP_LEARNINGRATE.domain.values:
          hparams = {
            HP_NUM_UNITS: num_units,
            HP_DROPOUT: dropout_rate,
            HP_LEARNINGRATE: learning_rate
          }
          run_name = "run-%d" % session_num
          print('--- Starting trial: %s' % run_name)
          #print({h.name: hparams[h] for h in hparams})
          waiting_model = model_fn('train', train_inputs, output_shape, hparams, HP_NUM_UNITS, HP_DROPOUT, HP_LEARNINGRATE)
          train_and_evaluate(waiting_model, data_set, log_dir, hparams, params, run_name)
          session_num += 1
