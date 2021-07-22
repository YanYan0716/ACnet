import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras


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
    inplanes=1,
    ratio=2,
    afilter=512,
    size=(28, 28),
    pfilter=8192,
    classes=config.CLASSES_NUM,
    firstStage=config.FIRST_SEAGE
)

# loss
loss = myLoss(alpha=1., betha=1.)

acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
training = CustomFit(model, acc_metric)
training.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1., momentum=0.9, ),
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
        print(flag, result)
    print('epoch:{epoch}')
# training.fit()