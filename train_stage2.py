import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

import config
from Loss import myLoss
from ACnet import ACnet
from Dataset import dataset
from CustomFit import CustomFit

# dataset
ds_train = dataset(config.DATA_PATH)
ds_test = dataset(config.DATA_TEST, train=False)

# model 第二阶段
model = ACnet(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    inplanes=1024,
    ratio=1,
    afilter=1024,
    size=(config.FTS_SIZE, config.FTS_SIZE),
    pfilter=1024,
    classes=config.CLASSES_NUM,
    firstStage=False
)

model.load_weights(config.LOAD_PATH)

# loss
loss = myLoss(alpha=1, betha=1.)
# optimizer
# 分段衰减学习率
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[20, 50, 70],
    values=[0.005, 0.005, 0.001, 0.0005]# [0.5, 0.1, 0.01, 0.005]
)
acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
training = CustomFit(model, acc_metric)
training.compile(
    optimizer=tfa.optimizers.SGDW(
        learning_rate=lr_schedule,
        momentum=0.9,
        weight_decay=5e-6
    ),
    loss=loss,
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

print('prepared second stage...')
print('==================================')
BEST_ACC = 0
for epoch in range(config.MAX_EPOCH):
    flag = 0
    training.acc_metric.reset_states()
    for (img, label) in ds_train:
        flag += 1
        result = training.train_step(data=(img, label))
        if flag % config.LOG_BATCH == 0:
            print(f'stage First: %s' % str(
                config.FIRST_SEAGE) + '    [max_epoch: %3d]' % config.MAX_EPOCH + '[epoch:%3d/' % (epoch + 1) \
                  + 'idx: %4d]' % flag + '[Loss:%.4f' % (result['loss'].numpy()) + ', ACC: %.2f]' % (
                      result['accuracy'].numpy()*100))

    if epoch % config.EVAL_EPOCH == 0:
        training.acc_metric.reset_states()
        for (img, label) in ds_test:
            result = training.test_step(data=(img, label))
        print(
            f'[testing ...]' + '[epoch:%3d/' % (epoch + 1) + '[Loss:%.4f' % (result['loss'].numpy()) + ',ACC: %.2f]' % (
                result['accuracy'].numpy()*100)
        )
        if result['accuracy'].numpy() > BEST_ACC:
            model.save_weights(config.SAVE_PATH, save_format='h5')
            BEST_ACC = result['accuracy'].numpy()
