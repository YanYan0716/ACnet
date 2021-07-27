import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

from Tree import BTree


def ACnet(
        input_shape=(448, 448, 3),
        inplanes=1,
        ratio=2,
        afilter=512,
        size=(28, 28),
        pfilter=8192,
        classes=200,
        firstStage=True,
):
    backbone = keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )
    if firstStage:
        backbone.trainable = False
    tree = BTree(inplanes, ratio, afilter, size, pfilter, classes)
    my_input = backbone.layers[0].input
    output = backbone.get_layer('block4_conv3').output
    output = keras.layers.Conv2D(
        afilter,
        1,
        1,
        activation='relu',
        kernel_initializer='random_normal'
    )(output)
    output = tree(output)
    all_model = keras.Model(inputs=my_input, outputs=output)
    return all_model


if __name__ == '__main__':
    img = tf.random.normal((2, 448, 448, 3))
    model = ACnet()
    labelPred1, labelPred2, labelPred3, labelPred4, AvgLabel = model(img)
    print(labelPred1.shape)
