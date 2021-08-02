import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy


import config


"""与paper不同，仅使用CE loss"""
class myLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, betha=1.):
        super(myLoss, self).__init__()
        self.loss = SparseCategoricalCrossentropy(from_logits=False)
        self.alpha = alpha
        self.betha = betha

    def __call__(self, y_true, y_pred, sample_weight=None):
        a = self.NLLLoss(y_true, y_pred[0]) * self.alpha
        b = self.NLLLoss(y_true, y_pred[1]) * self.alpha
        c = self.NLLLoss(y_true, y_pred[2]) * self.alpha
        d = self.NLLLoss(y_true, y_pred[3]) * self.alpha
        all_ = self.NLLLoss(y_true, y_pred[4]) * self.betha
        return all_+a+b+c+d

    def NLLLoss(self, y_true, y_pred):
        y_true = tf.one_hot(y_true, config.CLASSES_NUM)
        # y_pred = tf.nn.log_softmax(y_pred, axis=-1)
        y_pred = tf.math.log(y_pred+(1e-12))
        loss = tf.einsum('mn, mn -> mn', y_true, y_pred)
        loss = -tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        return loss

        

