import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy


"""与paper不同，仅使用CE loss"""
class myLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, betha=1.):
        super(myLoss, self).__init__()
        self.loss = SparseCategoricalCrossentropy(from_logits=True)
        self.alpha = alpha
        self.betha = betha

    def __call__(self, y_true, y_pred, sample_weight=None):
        a = self.loss_func(y_true, y_pred[0]) * self.alpha
        b = self.loss_func(y_true, y_pred[1]) * self.alpha
        c = self.loss_func(y_true, y_pred[2]) * self.alpha
        d = self.loss_func(y_true, y_pred[3]) * self.alpha
        all_ = self.loss_func(y_true, y_pred[4]) * self.betha
        return (a + b + c + d + all_)

    def loss_func(self, y_true, y_pred):
        y_pred = tf.nn.log_softmax(y_pred, axis=-1)
        loss = tf.einsum('mn, mn -> mn', y_true, y_pred)
        loss = -tf.reduce_mean(loss)
        return loss

        

