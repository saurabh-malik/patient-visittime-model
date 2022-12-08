import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np

class Custom_CE_Loss(Loss):
    # initialize instance attributes
    def __init__(self, alpha=1):
        super(Custom_CE_Loss, self).__init__()
        self.alpha = alpha
        
    # Compute loss
    def call(self, y_true, y_pred):
        ind = tf.range(y_pred.shape[1], dtype=tf.dtypes.int64)
        log_y_pred = tf.math.log(tf.math.subtract(1.0,y_pred))
        #log_y_pred = tf.math.log(y_pred)
        
        c = tf.math.argmax(y_true,
            axis=0,
            output_type=tf.dtypes.int64,
            name='true-label-Index'
        )

        indexes_w = tf.cast(tf.math.abs(tf.math.subtract(ind,c)),tf.float32)
        '''print(y_true)
        print(y_pred)
        print(ind)
        print(indexes_w)
        print('=====================')'''
        #indexes_w = tf.math.abs(tf.math.subtract(ind,c))
        #inv_ind = 1/tf.math.pow(indexes_w,1)
        inv_ind = tf.math.pow(indexes_w,0.1)
        y_true = tf.cast(y_true, tf.float32)
        inv_ind = tf.cast(inv_ind, tf.float32)
        elements = -tf.math.multiply_no_nan(x=inv_ind, y=log_y_pred)
        #velements = -tf.math.multiply_no_nan(x=log_y_pred, y=y_true)
        '''print(elements)
        print('=====================')'''

        loss = tf.reduce_mean(tf.reduce_sum(elements,axis=1))
        #vloss = tf.reduce_mean(tf.reduce_sum(velements,axis=1))
       

        return loss