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
    input_shape=(448, 448, 3),
    inplanes=1,
    ratio=2,
    afilter=512,
    size=(28, 28),
    pfilter=8192,
    classes=config.CLASSES_NUM)

# loss
loss = myLoss(alpha=1., betha=1.)

acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
training = CustomFit(model, acc_metric)
training.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.005, momentum=0.005),
    loss=loss
)

print('prepared ...')
print('==================================')
for epoch in range(config.MAX_EPOCH):
    for (img, label) in ds_train:
        training.train_step(data=(img, label))
    print('epoch:{epoch}')
# training.fit()