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
ds = dataset('../cifar/eee.csv', train=False)

# model 第一阶段
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

print('loading weights...')
model.load_weights('./model')
print('==================================')

acc = 0
for (img, label) in ds:
    pred = model(img)
    pred_index = tf.argmax(pred[-1], axis=-1)
    if label[0].numpy() == pred_index:
        acc += 1
print(acc)
