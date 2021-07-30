import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

from Tree import BTree


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
        self.conv = keras.layers.Conv2D(afilter, 1, 1, kernel_initializer='random_normal', activation='relu')
        if firstStage:
            self.backbone.trainable = False
        self.tree = BTree(inplanes, ratio, afilter, size, pfilter, classes)

    def call(self, inputs, training=None, mask=None):
        x_ = self.backbone(inputs)
        x = self.backbone.get_layer('conv2_block3_out').output
        x = self.conv(x)
        out = self.tree(x)
        return out

    def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
        super(acmodel, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        img, label = data
        # 前向传播
        with tf.GradientTape() as tape:
            pred = self.call(img, training=True)
            loss = self.loss(label, pred)

        training_vars = self.trainable_variables
        grads = tape.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(grads, training_vars))
        self.metric.update_state(label, pred[-1])
        return {'loss': loss, 'accuracy': self.metrics.result()}

    def test_step(self, data):
        img, label = data
        pred = self.call(img, training=False)
        loss = self.loss(label, pred)
        self.metrics.update_state(label, pred[-1])
        return {'loss': loss, 'accuracy': self.metrics.result()}


    def model(self):
        my_input = self.backbone.layers[0].input
        model = keras.Model(inputs=my_input, outputs=self.call(my_input))
        return model


if __name__ == '__main__':
    img = tf.random.normal((2, 448, 448, 3))
    model = acmodel()
    labelPred1, labelPred2, labelPred3, labelPred4, AvgLabel = model(img)
    print(labelPred1.shape)
    # model.save_weights('./123/111', save_format='h5')
    # print(model.layers[-1].trainable_variables)