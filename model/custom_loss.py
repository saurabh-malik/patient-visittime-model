import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np

class Custom_CE_Loss(Loss):
    # initialize instance attributes
    def __init__(self, gamma=1):
        super(Custom_CE_Loss, self).__init__()
        self.gamma = gamma
        
    # Compute loss
    def call(self, y_true, y_pred):
        ind = tf.range(y_pred.shape[1], dtype=tf.dtypes.int64)
        log_y_pred = tf.math.log(tf.math.subtract(1.0,y_pred))
        c = tf.math.argmax(y_true,
            axis=0,
            output_type=tf.dtypes.int64,
            name='true-label-Index'
        )
        indexes_w = tf.cast(tf.math.abs(tf.math.subtract(ind,c)),tf.float32)
        inv_ind = tf.math.pow(indexes_w,self.gamma)
        y_true = tf.cast(y_true, tf.float32)
        inv_ind = tf.cast(inv_ind, tf.float32)
        elements = -tf.math.multiply_no_nan(x=inv_ind, y=y_pred)
        return tf.reduce_mean(tf.reduce_sum(elements,axis=1))