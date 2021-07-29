import tensorflow as tf
DATA_PATH = '../input/cub-200-2011/train.csv'
DATA_TEST = '../input/cub-200-2011/test.csv'
SIZE = 512
IMG_SIZE = 448
BATCH_SIZE = 8
CLASSES_NUM = 200

FTS_SIZE = 28
MAX_EPOCH = 100
FIRST_SEAGE = True

LR_STEP = 100
INIT_LR = 0.02
LOG_BATCH = 100
EVAL_EPOCH = 5
LOAD_PATH = './model.h5'
SAVE_PATH = './model'
L2 = None# tf.keras.regularizers.l2(5e-6)