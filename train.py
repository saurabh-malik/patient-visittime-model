"""Train the model"""

import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from model.custom_loss import Custom_CE_Loss
from model.utils import Params
from model.utils import set_logger
from model.utils import df_to_dataset
from model.model_fn import model_fn
from model.encoder import get_normalization_layer
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
from model.training import train_and_evaluate
from model.feature_generator import encode_feature
from model.data_processing import load_data
from ConvertToLanguage import convert_to_unstructure
from tensorflow.keras.utils import plot_model
#import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('--data_version', default='V2',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/visits',
                    help="Directory containing the dataset")
parser.add_argument('--check_point', default=None,
                    help="Optional, previous training checkpoints")
parser.add_argument('--model_type', default='DNN',
                    help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--is_pretrain', default=0,
                    help="Optional, Define whether pretrain the language model. Only relevant with VLM model_type")
parser.add_argument('--BERT_Pretrain_Checkpoints', default="trained-models/pretraining_output/model.ckpt-20",
                    help="Optional, Provide pretrained BERT model check point")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    seed = 230
    tf.random.set_seed(seed)

    # Load the parameters from json file
    args = parser.parse_args()
    
    #Dataload
    datafile = 'visitdataclassification_'+args.data_version+'.csv'
    dataframe = load_data(args.data_dir, datafile)
    dataframe.head()

    #Model Type for Training
    model_type = args.model_type


    #Training for DNN (Baseline)
    if(model_type == 'DNN'):
        model_dir = 'experiments/base_model'
        json_path = os.path.join(model_dir, 'params.json')
        assert os.path.isfile(
            json_path), "No configuration file found at {}".format(json_path)
        params = Params(json_path)

        #load Configurable parameters
        batch_size = params.batch_size

        # Set the logger
        set_logger(os.path.join(model_dir, 'train.log'))

        # Create the data pipeline
        logging.info("Creating the datasets...")
        label_column = 'TotalTimeInWindow-15'

        #Drop Unused feature
        dataframe = dataframe.drop(columns=['TotalTimeInMin', 'VisitEnvelopeId', 'IsDropoffAppointment', 'NumberOfVisitsInProgress'])

        #Create Target Variable as One hot encode
        onehotecode = pd.get_dummies(dataframe[label_column], prefix='timewindow', sparse=True)
        y = onehotecode.values
        output_shape = y.shape[1]
        
        #Split Data into 60/20/20
        train_x, hold_x, train_y, hold_y = train_test_split(dataframe, y, test_size=0.4)
        val_x, test_x, val_y, test_y = train_test_split(hold_x, hold_y, test_size=0.5)

        #Conversion to tensorflow datase
        ds = df_to_dataset(dataframe, y, batch_size=batch_size)
        train_ds = df_to_dataset(train_x, train_y, batch_size=batch_size)
        val_ds = df_to_dataset(val_x, val_y, shuffle=False, batch_size=batch_size)
        test_ds = df_to_dataset(test_x, test_y, shuffle=False, batch_size=batch_size)

        ## ToDo Move Feature creation into seprate module. Done
        #Features
        numerical_features = [ 'AgeInMonths','Weight']
        categorical_features = [ 'AnimalClass', 'Sex', 'Day','AnimalBreed', 'AppointmentTypeName','AppointmentReasons']
        
        #Encoded features.
        all_inputs, encoded_features = encode_feature(numerical_features, categorical_features, ds)
        train_inputs = {'all_inputs': all_inputs, 'encoded_features': encoded_features}

        logging.info("Creating the model...")
        #Hyperparameter
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([params.num_units]))
        HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([params.dropout_rate]))
        HP_LEARNINGRATE = hp.HParam('learning_rate', hp.Discrete([params.learning_rate]))
        METRIC_ACCURACY = 'accuracy'
        hparams = {
            HP_NUM_UNITS: params.num_units,
            HP_DROPOUT: params.dropout_rate,
            HP_LEARNINGRATE: params.learning_rate
        }
       
        data_set = {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
        }
        log_dir = model_dir + '/logs/fit'
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams_config(
                hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_LEARNINGRATE],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
            )


        waiting_model = model_fn('train', train_inputs, output_shape, hparams, HP_NUM_UNITS, HP_DROPOUT, HP_LEARNINGRATE)
        run_name = 'final run'

        train_and_evaluate(waiting_model, data_set, log_dir, hparams, params, run_name)

    else if model_type == 'VLM':
        #Model Type is Language Model (VLM)
        model_dir = 'experiments/base_model'
        data_dir = args.data_dir

        json_path = os.path.join(model_dir, 'params.json')
        assert os.path.isfile(
            json_path), "No configuration file found at {}".format(json_path)
        params = Params(json_path)

        # Set the logger
        set_logger(os.path.join(model_dir, 'train.log'))
        log_dir = ''

        #Data Pipeline
        train_x, hold_x = train_test_split(dataframe, test_size=0.20)


        #Data Conversion Structure to Unstructured text
        #Train and Validation text
        text_data_dir = data_dir+'language_model/'
        # Test texts
        if(os.path.exists(text_data_folder+'train') == False):
            convert_to_unstructure(train_x, text_data_dir)
        if(os.path.exists(text_data_folder+'test') == False):
            convert_to_unstructure(hold_x, text_data_dir, False)

        #Split is 0.25. i.e 0.20 of overall validation set from already splited 0.80 train data
        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            text_data_dir+'train',
            batch_size=params.batch_size,
            validation_split=0.25,
            subset='training',
            seed=seed)
        val_ds = tf.keras.utils.text_dataset_from_directory(
            text_data_dir+'train',
            batch_size=params.batch_size,
            validation_split=0.25,
            subset='validation',
            seed=seed)
        test_ds = tf.keras.utils.text_dataset_from_directory(
            'data/visits/test',
            batch_size=params.batch_size)

        train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        data_set = {
                'train_ds': train_ds,
                'val_ds': val_ds,
                'test_ds': test_ds,
            }

        class_names = raw_train_ds.class_names

        #BERT Model for Finetune
        bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
        map_name_to_handle = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_base/2',
            'electra_small':
                'https://tfhub.dev/google/electra_small/2',
            'electra_base':
                'https://tfhub.dev/google/electra_base/2',
            'experts_pubmed':
                'https://tfhub.dev/google/experts/bert/pubmed/2',
            'experts_wiki_books':
                'https://tfhub.dev/google/experts/bert/wiki_books/2',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
        }
        map_model_to_preprocess = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
            'electra_small':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'electra_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_pubmed':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_wiki_books':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        }
        tfhub_handle_encoder = map_name_to_handle[bert_model_name]
        tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

        print(f'BERT model selected           : {tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

        bert_model = hub.KerasLayer(tfhub_handle_encoder)

        bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

        pretrain_checkpoints = args.BERT_Pretrain_Checkpoints

        # Configure Hyperparameter for Language Model
        HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([params.dropout_rate]))
        HP_LEARNINGRATE = hp.HParam('learning_rate', hp.Discrete([params.learning_rate]))
        METRIC_ACCURACY = 'accuracy'
        hparams = {
              HP_DROPOUT: params.dropout_rate,
              HP_LEARNINGRATE: params.learning_rate
          }

        #Todo Remove the build_Pretrain_BERT_model. Not needed
        def build_Pretrain_BERT_model():
          text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
          preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
          encoder_inputs = preprocessing_layer(text_input)
          encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
          outputs = encoder(encoder_inputs)
          net = outputs['pooled_output']
          model = tf.keras.Model(text_input, net)
          checkpoint = tf.train.Checkpoint(model)
          checkpoint.restore(pretrain_checkpoints)
          return model

        #ToDo Move into model_fn class
        def build_vlm_model(classes,params, HP_DROPOUT, pretrain_checkpoints):
            drop_out = params[HP_DROPOUT]
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
            preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
            encoder_inputs = preprocessing_layer(text_input)
            encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
            outputs = encoder(encoder_inputs)
            net = outputs['pooled_output']
            net = tf.keras.layers.Dropout(drop_out)(net)
            net = tf.keras.layers.Dense(classes, activation="softmax", name='classifier')(net)
            model = tf.keras.Model(text_input, net)
            checkpoint = tf.train.Checkpoint(model)
            checkpoint.restore(pretrain_checkpoints)
            return model

        language_model = build_vlm_model(len(class_names), hparams, HP_DROPOUT, pretrain_checkpoints)

        
        #Defining the graph operation
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,ignore_class=None,
            name='sparse_categorical_crossentropy')
        #loss = Custom_CE_Loss(gamma=0.1)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * params.num_epochs
        num_warmup_steps = int(0.1*num_train_steps)
        learning_rate = hparams[HP_LEARNINGRATE] 
        optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        language_model.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=metrics)

        #tf.keras.utils.plot_model(language_model)
        print(f'Training model with {tfhub_handle_encoder}')

        #checkpoint
        posttraining_checkpoint_path = "trained-models/language/training_VLM_2/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        #Train and evaluate
        run_name+='_lr-'+str(learning_rate)+'_ep-'+str(params.num_epochs)
        train_and_evaluate(language_model, data_set, log_dir, hparams, params, run_name)

    else:
        print('Provide model_type either DNN or VLM')