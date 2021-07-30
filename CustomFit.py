import tensorflow as tf
import tensorflow.keras as keras


class CustomFit(keras.Model):
    def __init__(self, model, acc_metric):
        super(CustomFit, self).__init__()
        self.model = model
        self.acc_metric = acc_metric

    def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        img, label = data
        # 前向传播
        with tf.GradientTape() as tape:
            pred = self.model(img, training=True)
            loss = self.loss(label, pred)

        training_vars = self.model.trainable_variables
        grads = tape.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(grads, training_vars))
        self.acc_metric.update_state(label, pred[-1])
        return {'loss': loss, 'accuracy': self.acc_metric.result()}

    def test_step(self, data):
        img, label = data
        pred = self.model(img, training=False)
        loss = self.loss(label, pred)
        self.acc_metric.update_state(label, pred[-1])
        return {'loss': loss, 'accuracy': self.acc_metric.result()}
