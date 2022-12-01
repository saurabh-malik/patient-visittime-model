import tensorflow as tf
from tensorflow.keras.losses import Loss

class Custom_CE_Loss(Loss):
    # initialize instance attributes
    def __init__(self, alpha=1):
        super(Custom_CE_Loss, self).__init__()
        self.alpha = alpha
        
    # Compute loss
    def call(self, y_true, y_pred):
        log_y_pred = tf.math.log(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        elements = -tf.math.multiply_no_nan(x=log_y_pred, y=y_true)
        return tf.reduce_mean(tf.reduce_sum(elements,axis=1))