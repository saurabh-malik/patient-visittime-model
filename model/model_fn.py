"""Define the model."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp

def build_model(is_training, output_shape, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    #Merge feature inputs into one vector
    all_features = tf.keras.layers.concatenate(inputs['encoded_features'])
    print(params)
    #Create HParams as model hyperparameter
    hp_dropout = params['HP_DROPOUT']
    hp_num_units = params['HP_NUM_UNITS']
    
    x = tf.keras.layers.Dense(hp_num_units, activation="relu")(all_features)
    x=tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(292, activation="relu")(x)
    x=tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(584, activation="relu")(x)
    x=tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(2336, activation="relu")(x)
    x=tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(4672, activation="relu")(x)
    x=tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(1168, activation="relu")(x)
    x=tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(584, activation="relu")(x)
    x=tf.keras.layers.Dropout(hp_dropout)(x)
    #x = tf.keras.layers.Dense(5000, activation="relu")(x)
    #x=tf.keras.layers.Dropout(0.7)(x)
    #x = tf.keras.layers.Dense(1000, activation="relu")(x)
    #x=tf.keras.layers.Dropout(0.6)(x)
    #x = tf.keras.layers.Dense(1200, activation="relu")(x)
    #x=tf.keras.layers.Dropout(0.7)(x)
    #x = tf.keras.layers.Dense(300, activation="relu")(x)
    #x=tf.keras.layers.Dropout(0.7)(x)
    #x = tf.keras.layers.Dense(190, activation="relu")(x)
    #x=tf.keras.layers.Dropout(0.7)(x)
    
    output = tf.keras.layers.Dense(output_shape, activation="softmax")(x)

    model = keras.Model(inputs['all_inputs'], output)

    return model


def model_fn(mode, inputs, output_shape, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train') 

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    model = build_model(is_training, output_shape, inputs, params)

    # Define loss and accuracy
    loss = loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    accuracy = ["accuracy"]
    optimizer = 'adam'

    #learning rate hyperparameter
    hp_learningrate = params['HP_LEARNINGRATE']

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

