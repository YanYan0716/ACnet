import tensorflow as tf
DATA_PATH = '../input/cifar10/cifar/train.csv'
DATA_TEST = '../input/cifar10/cifar/test.csv'
SIZE = 256
IMG_SIZE = 224
BATCH_SIZE = 4
CLASSES_NUM = 10

FTS_SIZE = 14
MAX_EPOCH = 100
FIRST_SEAGE = True

LR_STEP = 100
INIT_LR = 0.02
LOG_BATCH = 100
EVAL_EPOCH = 5
LOAD_PATH = './model.h5'
SAVE_PATH = './model'
L2 = None# tf.keras.regularizers.l2(5e-6)