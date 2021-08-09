import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

import config
from Loss import myLoss
from Dataset import dataset
from CustomFit import CustomFit
from acmodel import acmodel
from ACnet import ACnet

# dataset
ds_train = dataset(config.DATA_PATH)
ds_test = dataset(config.DATA_TEST, train=False)

# model 第一阶段
# model = acmodel(
#     input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
#     inplanes=512,
#     ratio=1,
#     afilter=512,
#     size=(config.FTS_SIZE, config.FTS_SIZE),
#     pfilter=512,
#     classes=config.CLASSES_NUM,
#     firstStage=config.FIRST_SEAGE
# ).model()
model = ACnet(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    inplanes=512,
    ratio=1,
    afilter=512,
    size=(config.FTS_SIZE, config.FTS_SIZE),
    pfilter=512,
    classes=config.CLASSES_NUM,
    firstStage=config.FIRST_SEAGE
)

# loss
myloss = myLoss(alpha=1, betha=1.)
# optimizer
# 分段衰减学习率
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[20, 50, 70],
    values=[0.05, 0.01, 0.001, 0.0005]  # [0.5, 0.1, 0.01, 0.005]
)

train_metric = keras.metrics.SparseCategoricalAccuracy()
val_metric = keras.metrics.SparseCategoricalAccuracy()
optim = tf.optimizers.SGD(learning_rate=lr_schedule)

# model.compile(
#     optimizer=optim,
#     loss=myloss,
#     metrics=['accuracy'],
# )
print('prepared first stage...')
print('==================================')
BEST_ACC = 0

# model.fit(ds_train, epochs=5, verbose=2, validation_data=ds_test)

for epoch in range(config.MAX_EPOCH):
    flag = 0
    train_metric.reset_states()
    for (img, label) in ds_train:
        flag += 1
        with tf.GradientTape() as tape:
            y_pred = model(img, training=True)
            loss = myloss(label, y_pred)
        gradients = tape.gradient(loss, model.trainable_weights)
        optim.apply_gradients(zip(gradients, model.trainable_weights))
        train_metric.update_state(label, y_pred)

        if flag % config.LOG_BATCH == 0:
            print(f'stage First: %s' % str(
                config.FIRST_SEAGE) + '    [max_epoch: %3d]' % config.MAX_EPOCH + '[epoch:%3d/' % (epoch + 1) \
                  + 'idx: %4d]' % flag + '[Loss:%.4f' % (loss.numpy()) + ', ACC: %.2f]' % (
                          train_metric.result().numpy() * 100))

    train_metric.reset_states()
    if (epoch + 1) % config.EVAL_EPOCH == 0:
        for (img, label) in ds_test:
            y_pred = model(img, training=False)
            val_metric.update_state(label, y_pred)
        print(
            f'[testing ...]' + '[epoch:%3d/' % (epoch + 1) + ',ACC: %.2f]' % (val_metric.result().numpy() * 100)
        )
        if val_metric.result().numpy() > BEST_ACC:
            model.save_weights(config.SAVE_PATH, save_format='h5')
            BEST_ACC = val_metric.result().numpy()
        val_metric.reset_states()
