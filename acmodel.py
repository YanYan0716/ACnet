import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

from Tree import BTree
import config

class acmodel(keras.Model):
    def __init__(self,
                 input_shape=(448, 448, 3),
                 inplanes=1,
                 ratio=2,
                 afilter=512,
                 size=(28, 28),
                 pfilter=8192,
                 classes=200,
                 firstStage=True,
):
        super(acmodel, self).__init__()
        self.backbone = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
        )
        self.conv = keras.layers.Conv2D(
            afilter,
            1,
            1,
            kernel_initializer=config.CONV_INIT,
            activation='relu'
        )
        self.tree = BTree(inplanes, ratio, afilter, size, pfilter, classes)
        if firstStage:
            self.backbone.trainable = False
            self.conv.trainable = True
            self.tree.trainable = True

    def call(self, inputs, training=None, mask=None):
        x_ = self.backbone(inputs)
        x = self.backbone.get_layer('conv2_block3_out').output
        x = self.conv(x)
        out = self.tree(x)
        return out

    def model(self):
        my_input = self.backbone.layers[0].input
        model = keras.Model(inputs=my_input, outputs=self.call(my_input))
        return model


if __name__ == '__main__':
    # img = tf.random.normal((2, 448, 448, 3))
    model = acmodel().model()
    # labelPred1, labelPred2, labelPred3, labelPred4, AvgLabel = model(img)
    # print(labelPred1.shape)
    # model.save_weights('./123/111', save_format='h5')
    # model.save('123')
    # print(model.layers[-1].trainable_variables)