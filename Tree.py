import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras


from Route import Route
from Attention import Attention
from LabelPred import LabelPred


class BTree(keras.layers.Layer):
    def __init__(self, inplanes, ratio, afilter, size, pfilter, classes):
        super(BTree, self).__init__()
        self.route1_L1 = Route(inplanes, ratio)
        self.route1_L2 = Route(inplanes, ratio)
        self.route2_L2 = Route(inplanes, ratio)

        self.att_1 = Attention(afilter, size)
        self.att_2 = Attention(afilter, size)
        self.att_3 = Attention(afilter, size)

        self.att_4 = Attention(afilter, size)
        self.att_6 = Attention(afilter, size)
        self.att_8 = Attention(afilter, size)

        self.att_5 = Attention(afilter, size)
        self.att_7 = Attention(afilter, size)
        self.att_9 = Attention(afilter, size)

        self.p_1 = LabelPred(pfilter, classes)
        self.p_2 = LabelPred(pfilter, classes)
        self.p_3 = LabelPred(pfilter, classes)
        self.p_4 = LabelPred(pfilter, classes)

    def call(self, inputs):
        theta1_l1 = self.route1_L1(inputs)
