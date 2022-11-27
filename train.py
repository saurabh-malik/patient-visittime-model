"""Train the model"""

import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot

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


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
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
    #data_dir = args.data_dir
    #train_data_dir = os.path.join(data_dir, "train_signs")
    #dev_data_dir = os.path.join(data_dir, "dev_signs")

    #------------------------------------------------------------------------
    #Update

    #DataLoad & Split
    #Dataload
    file = args.data_dir + "/visitdataclassification-4300.csv"
    dataframe = pd.read_csv(file)
    dataframe.head()

    #Create Target Variable as One hot encode
    #onehotecode = pd.get_dummies(dataframe['TotalTimeInWindow-15'], prefix='timewindow', sparse=True)
    onehotecode = pd.get_dummies(dataframe['TotalTimeInWindow-15'], prefix='timewindow', sparse=True)
    y = onehotecode.values
    output_shape = y.shape[1]

    print('output_shape: ', output_shape)

    #Drop Unused feature
    dataframe = dataframe.drop(columns=['TotalTimeInMin', 'VisitEnvelopeId', 'IsDropoffAppointment', 'TotalTimeInWindow-30', 'TotalTimeInWindow-15'])

    #Split Data
    
    train_x, hold_x, train_y, hold_y = train_test_split(dataframe, y, test_size=0.40)
    val_x, test_x, val_y, test_y = train_test_split(hold_x, hold_y, test_size=0.50)

    #train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])

    batch_size = params.batch_size
    train_ds = df_to_dataset(train_x, train_y, batch_size=batch_size)
    val_ds = df_to_dataset(val_x, val_y, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test_x, test_y, shuffle=False, batch_size=batch_size)


    #Features
    numerical_features = ['AgeInMonths', 'Weight', 'NumberOfVisitsInProgress']
    categorical_features = [ 'AnimalClass', 'Sex', 'Day','AnimalBreed', 'AppointmentTypeName','AppointmentReasons']
    encoded_features = []
    all_inputs = []

    #Numerical features.
    for header in numerical_features:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    #Categorical features.
    for header in categorical_features:
      categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
      encoding_layer = get_category_encoding_layer(name=header,
                                                   dataset=train_ds,
                                                   dtype='string',
                                                   max_tokens=20)
      encoded_categorical_col = encoding_layer(categorical_col)
      all_inputs.append(categorical_col)
      encoded_features.append(encoded_categorical_col)


    train_inputs = {'all_inputs': all_inputs, 'encoded_features': encoded_features}

    logging.info("Creating the model...")
    #train_inputs()
    #Hyperparameter
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([params.num_units]))
    HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([params.dropout_rate]))
    HP_LEARNINGRATE = hp.HParam('learning_rate', hp.Discrete([params.learning_rate]))
    METRIC_ACCURACY = 'accuracy'
    hparams = {
          'HP_NUM_UNITS': params.num_units,
          'HP_DROPOUT': params.dropout_rate,
          'HP_LEARNINGRATE': params.learning_rate
      }

    #print({h.name: hparams[h] for h in hparams})

    waiting_model = model_fn('train', train_inputs, output_shape, hparams)
    #eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    #train and evaluate model
    train_and_evaluate(waiting_model, args.data_dir, hparams)
