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

# model
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
loss = myLoss(alpha=1., betha=1.)
# optimizer
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    config.INIT_LR,
    decay_steps=config.MAX_EPOCH,
    decay_rate=0.96,
    staircase=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
training = CustomFit(model, acc_metric)
training.compile(
    optimizer=tfa.optimizers.SGDW(
        learning_rate=lr_schedule,
        momentum=0.9,
        weight_decay=5e-6
    ),
    loss=loss,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

print('prepared ...')
print('==================================')
for epoch in range(config.MAX_EPOCH):
    flag = 0
    for (img, label) in ds_train:
        flag += 1
        result = training.train_step(data=(img, label))
        if flag % 20 == 0:
            print(flag, result)
    print('epoch:{}'.format(epoch+1))
# training.fit()