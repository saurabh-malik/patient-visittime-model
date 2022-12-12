"""Define the model."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp
from model.custom_loss import Custom_CE_Loss

def build_model(is_training, output_shape, inputs, params, HP_NUM_UNITS, HP_DROPOUT,):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        output_shape: (int)
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """

    #Create HParams as model hyperparameter
    hp_dropout = params[HP_DROPOUT]
    hp_num_units = params[HP_NUM_UNITS]
    
    #Merge feature inputs into one vector
    all_features = tf.keras.layers.concatenate(inputs['encoded_features'])

    x = tf.keras.layers.Dense(hp_num_units, activation="relu")(all_features)
    x=tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.Dense(2500, activation="relu")(x)
    x=tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(1500, activation="relu")(x)
    x=tf.keras.layers.Dropout(0.1)(x)

    output = tf.keras.layers.Dense(output_shape, activation="softmax")(x)

    model = keras.Model(inputs['all_inputs'], output)

    return model


def model_fn(mode, inputs, output_shape, params, HP_NUM_UNITS, HP_DROPOUT, HP_LEARNINGRATE, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        output_shape: (int)
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train') 

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    model = build_model(is_training, output_shape, inputs, params, HP_NUM_UNITS, HP_DROPOUT,)
    alpha=0.5
    # Define loss and accuracy
    #loss = Custom_CE_Loss(alpha)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, name='categorical_crossentropy' )
    accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32)
    #accuracy = tf.keras.metrics.Recall(name = 'recall')
    optimizer = 'adam'

    #learning rate hyperparameter
    hp_learningrate = params[HP_LEARNINGRATE]

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(hp_learningrate)

    #if is_training:
        #global_step = tf.train.get_or_create_global_step()
        #if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                #train_op = optimizer.minimize(loss, global_step=global_step)
        #else:
            #train_op = optimizer.minimize(loss, global_step=global_step)

    model.compile(optimizer = optimizer, loss = loss, metrics = accuracy)

    return model

