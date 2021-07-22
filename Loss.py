import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy


"""与paper不同，仅使用CE loss"""
class myLoss(keras.losses.Loss):
    def __init__(self, alpha=1., betha=1.):
        super(myLoss, self).__init__()
        self.loss = SparseCategoricalCrossentropy()
        self.alpha = alpha
        self.betha = betha

    def __call__(self, y_true, y_pred, sample_weight=None):
        a = self.loss(y_true, y_pred[0]) * self.alpha
        b = self.loss(y_true, y_pred[1]) * self.alpha
        c = self.loss(y_true, y_pred[2]) * self.alpha
        d = self.loss(y_true, y_pred[3]) * self.alpha
        all = self.loss(y_true, y_pred[4]) * self.betha
        
        return a+b+c+d+all
