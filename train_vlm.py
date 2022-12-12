import os
import shutil
import pandas as pd
import argparse

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt

from official.nlp import optimization  # to create AdamW optimizer
from string import Template
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

from model.data_processing import load_data
from model.conversion import convert_to_unstructure
from model.conversion import convert_to_unstructure_for_pretraining
from model.utils import print_my_examples
from model.custom_loss import Custom_CE_Loss
from model.training import train_and_evaluate
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")

if __name__ == '__main__':
    AUTOTUNE = tf.data.AUTOTUNE
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    seed = 42
    log_dir = 'experiments/language_model'
    run_name = 'post_pretune_final'

    #Load Data
    dataframe = load_data("data/visits", "/visitdataclassification-4300.csv")
    dataframe.head()
    train_x, hold_x = train_test_split(dataframe, test_size=0.20)


    #data for pretrain
    convert_to_unstructure_for_pretraining(dataframe)
    processedFolder = "data/visits/train"
    if(os.path.exists(processedFolder) == False):
        convert_to_unstructure(train_x)
        convert_to_unstructure(hold_x, False)

    #
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'data/visits/train',
        batch_size=params.batch_size,
        validation_split=0.25,
        subset='training',
        seed=seed)

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        'data/visits/train',
        batch_size=params.batch_size,
        validation_split=0.25,
        subset='validation',
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        'data/visits/test',
        batch_size=params.batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_set = {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
        }

    for text_batch, label_batch in train_ds.take(1):
      for i in range(10):
        print(f'Review: {text_batch.numpy()[i]}')
        label = label_batch.numpy()[i]
        print(f'Label : {label} ({class_names[label]})')

    #Select Model
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

    checkpoint_path = "trained-models/pretraining_output/model.ckpt-20"

    # Configure Hyperparameter for Language Model
    HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([params.dropout_rate]))
    HP_LEARNINGRATE = hp.HParam('learning_rate', hp.Discrete([params.learning_rate]))
    METRIC_ACCURACY = 'accuracy'
    hparams = {
          HP_DROPOUT: params.dropout_rate,
          HP_LEARNINGRATE: params.learning_rate
      }

    def build_BERT_model():
      text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
      preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
      encoder_inputs = preprocessing_layer(text_input)
      encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
      outputs = encoder(encoder_inputs)
      net = outputs['pooled_output']
      model = tf.keras.Model(text_input, net)
      checkpoint = tf.train.Checkpoint(model)
      checkpoint.restore(checkpoint_path)
      #model.load_weights(checkpoint_path)
      return model

    def build_classifier_model(classes,params, HP_DROPOUT):
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
        checkpoint.restore(checkpoint_path)
        return model


    #bert_model = build_BERT_model()
    #save model
    #bert_model.save('bertmodel')

    language_model = build_classifier_model(len(class_names), hparams, HP_DROPOUT)
    #bert_raw_result = language_model(tf.constant(text_test))
    #print(tf.sigmoid(bert_raw_result))

    #tf.keras.utils.plot_model(language_model)
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,ignore_class=None,
        name='sparse_categorical_crossentropy')
    
    #loss = Custom_CE_Loss(gamma=0.1)
    #metrics = tf.keras.metrics.Recall()
    metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * params.num_epochs
    num_warmup_steps = int(0.1*num_train_steps)

    #init_lr = 1e-3
    learning_rate = hparams[HP_LEARNINGRATE] 
    optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    language_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    print(f'Training model with {tfhub_handle_encoder}')

    #checkpoint
    posttraining_checkpoint_path = "trained-models/language/training_CustomLoss_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #One Sample Testing
    examples = [
        #'A 29 months old, 50 lb, Male Dalmatian-Canine is checked-in on Monday in hospital for Exam Annual due to Vaccines.',  # this is the same sentence tried earlier
        'A 45 months old, 72 lb, Male Labrador Retriever Mix-Canine is checked-in on Tuesday in hospital for Technician Appointment due to Exam.'
    ]
    original_results = tf.sigmoid(language_model(tf.constant(examples)))

    #Train and evaluate
    run_name+='_lr-'+str(learning_rate)+'_ep-'+str(params.num_epochs)
    #language_model.summary()
    train_and_evaluate(language_model, data_set, log_dir, hparams, params, run_name)

    post_finetunning_results = tf.sigmoid(language_model(tf.constant(examples)))



    #Uncomment below if you want to visualize the results pre and post training
    '''
    print('-----Model prediction-----')
    print('Results from the model without training:')
    print(original_results)
    print_my_examples(examples, original_results)
    plt.plot(original_results[0], linestyle = 'dotted')
    plt.title('Visit Time Window before Fine Tunning distribution')
    plt.show()

    print('Results from the saved model:')
    print(post_finetunning_results)
    print_my_examples(examples, post_finetunning_results)
    plt.plot(post_finetunning_results[0], linestyle = 'dotted')
    plt.title('Visit Time Window Post Fine Tunning distribution')
    plt.show()
    '''