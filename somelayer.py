"""关于pytorch和tf的异同"""
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf

# batchnorm
a = np.random.normal(size=(1, 3, 4, 4))
torch_a = torch.from_numpy(a).to(torch.float32)
torchBN = torch.nn.BatchNorm2d(3)
tf_a = tf.convert_to_tensor(a, dtype=tf.float32)
tf_a = tf.transpose(tf_a, [0, 2, 3, 1])
inputs = tf.keras.Input(shape=(4, 4, 3))
output = tf.keras.layers.BatchNormalization(
    epsilon=1e-5,
    momentum=0.9,
    axis=-1,
    center=True,
    scale=True,
)(inputs)
tfBN = tf.keras.Model(inputs, outputs=output)
torch_out = torchBN(torch_a)
tf_out = tfBN(tf_a, training=True)
print('----------')
print(torch_out[0][0])
print('------------')
print(tf_out[0, :, :, 0])

# global pooling
a = np.random.normal(size=(1, 3, 4, 4))
torch_a = torch.from_numpy(a).to(torch.float32)
tf_a = tf.convert_to_tensor(a, dtype=tf.float32)
tf_a = tf.transpose(tf_a, [0, 2, 3, 1])

avgpool_torch = torch.nn.AdaptiveAvgPool2d((1, 1))
avgpool_tf = tf.keras.layers.GlobalAveragePooling2D()

y1 = avgpool_torch(torch_a)
y2 = avgpool_tf(tf_a)
print(y1)
print(y2)

# layernorm
a = np.random.normal(size=(1, 3, 4, 4))

torch_a = torch.from_numpy(a).to(torch.float32)
tf_a = tf.convert_to_tensor(a, dtype=tf.float32)
tf_a = tf.transpose(tf_a, [0, 2, 3, 1])

torchL = torch.nn.LayerNorm([3, 4, 4])
tfL = tf.keras.layers.LayerNormalization(
    epsilon=1e-5,
    axis=[1, 2, 3]
)

y1 = torchL(torch_a)
y2 = tfL(tf_a)
print('----------')
print(y1[0][0])
print('------------')
print(y2[0, :, :, 0])

#l2 norm
a=np.random.normal(size=(1,3,4,4))
torch_a = torch.from_numpy(a).to(torch.float32)
tf_a = tf.convert_to_tensor(a, dtype=tf.float32)
tf_a = tf.transpose(tf_a, [0, 2, 3, 1])

torchL = lambda x: F.normalize(x, dim=1)
tfL = tf.math.l2_normalize

y1=torchL(torch_a)
y2=tfL(tf_a, axis=-1)
print('----------')
print(y1[0][0])
print('------------')
print(y2[0,:,:,0])
