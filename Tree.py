import os

from tensorflow.keras import regularizers

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras


from Route import Route
from Attention import Attention
from LabelPred import LabelPred
import config


class BTree(keras.layers.Layer):
    def __init__(self, inplanes, ratio, afilter, size, pfilter, classes=config.CLASSES_NUM):
        super(BTree, self).__init__()
        self.route1_L1 = Route(inplanes, ratio, channel=afilter)
        self.route1_L2 = Route(inplanes, ratio, channel=afilter)
        self.route2_L2 = Route(inplanes, ratio, channel=afilter)

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
        """为了清楚搭建树结构，未使用任何循环简化代码，本树结构完全按照论文figure2实现"""
        route_result = [[0, 0], [0, 0, 0, 0]]
        features_result = [[None], [None, None], [None, None, None, None]]

        # 第一层Route
        left_prob1_l1 = self.route1_L1(inputs)
        right_prob1_l1 = 1 - left_prob1_l1
        route_result[0][0], route_result[0][1] = left_prob1_l1, right_prob1_l1

        features_result[0][0] = inputs
        # 第一层attention
        left_fts1_l1 = self.att_3(self.att_1(inputs))
        right_fts1_l1 = self.att_2(inputs)
        features_result[1][0], features_result[1][1] = left_fts1_l1, right_fts1_l1

        # 第二层Route
        left_prob1_l2 = self.route1_L2(features_result[1][0])
        right_prob1_l2 = 1 - left_prob1_l2
        left_prob2_l2 = self.route2_L2(features_result[1][1])
        right_prob2_l2 = 1 - left_prob2_l2
        route_result[1][0], route_result[1][1] = left_prob1_l2*route_result[0][0], right_prob1_l2*route_result[0][0]
        route_result[1][2], route_result[1][3] = left_prob2_l2*route_result[0][1], right_prob2_l2*route_result[0][1]

        # 第二层attention
        left_fts1_l2 = self.att_8(self.att_4(features_result[1][0]))
        right_fts1_l2 = self.att_6(features_result[1][0])
        features_result[2][0], features_result[2][1] = left_fts1_l2, right_fts1_l2

        left_fts2_l2 = self.att_9(self.att_5(features_result[1][1]))
        right_fts2_l2 = self.att_7(features_result[1][1])
        features_result[2][2], features_result[2][3] = left_fts2_l2, right_fts2_l2

        # label pred
        labelPred1 = self.p_1(features_result[2][0])
        AvgLabel = (labelPred1*route_result[1][0])

        labelPred2 = self.p_2(features_result[2][1])
        AvgLabel += (labelPred2*route_result[1][1])

        labelPred3 = self.p_3(features_result[2][2])
        AvgLabel += (labelPred3*route_result[1][2])

        labelPred4 = self.p_4(features_result[2][3])
        AvgLabel += (labelPred4*route_result[1][3])

        return labelPred1, labelPred2, labelPred3, labelPred4, AvgLabel


if __name__ == '__main__':
    img = tf.random.normal((4, 14, 14, 1024))
    tree = BTree(inplanes=512, ratio=2, afilter=1024, size=(14, 14), pfilter=8192, classes=200)
    # y = tree(img)
    # print('ok')