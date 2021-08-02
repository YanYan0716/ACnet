import tensorflow as tf
DATA_PATH = '../input/cifar10/cifar/train.csv'
DATA_TEST = '../input/cifar10/cifar/test.csv'
SIZE = 130
IMG_SIZE = 112
BATCH_SIZE = 24
CLASSES_NUM = 10

FTS_SIZE = 28
MAX_EPOCH = 100
FIRST_SEAGE = True

LR_STEP = 100
INIT_LR = 0.02
LOG_BATCH = 100
EVAL_EPOCH = 1
LOAD_PATH = './model.h5'
SAVE_PATH = './model.h5'
L2 = tf.keras.regularizers.l2(5e-6)