import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

'''
reference: 
    https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
paper:
    https://arxiv.org/pdf/1904.11492.pdf
'''
class GlobalContext(keras.layers.Layer):
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(GlobalContext, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = keras.layers.Conv2D(filters=1, kernel_size=1,)
            self.softmax = keras.layers.Softmax(axis=1)
        else:
            self.avg_pool = keras.layers.AveragePooling2D()
        if 'channel_add' in fusion_types:
            self.channel_add_conv = keras.Sequential([
                keras.layers.Conv2D(self.planes, kernel_size=1),
                keras.layers.LayerNormalization(axis=[1, 2, 3]),
                keras.layers.ReLU(),
                keras.layers.Conv2D(self.inplanes, kernel_size=1)
            ])
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = keras.Sequential([
                keras.layers.Conv2D(self.planes, kernel_size=1),
                keras.layers.LayerNormalization(axis=[1, 2, 3]),
                keras.layers.ReLU(),
                keras.layers.Conv2D(self.inplanes, kernel_size=1)
            ])
        else:
            self.channel_mul_conv = None

        # self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            pass

    def spatical_pool(self, x):
        batch, height, width, channel = x.shape
        if self.pooling_type == 'att':
            input_x = x
            input_x = tf.reshape(input_x, (batch, height*width, channel))
            input_x = tf.transpose(input_x, [0, 2, 1])

            context_mask = self.conv_mask(x)
            context_mask = tf.reshape(context_mask, (batch, height*width))
            context_mask = self.softmax(context_mask)
            context_mask = tf.expand_dims(context_mask, axis=-1)

            context = tf.matmul(input_x, context_mask)
            context = tf.reshape(context, (batch, 1, 1, channel))
        else:
            context = self.avg_pool(x)

        return context

    def call(self, inputs, **kwargs):
        context = self.spatical_pool(inputs)

        out = inputs
        if self.channel_mul_conv is not None:
            channel_mul_term = keras.activations.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class Route(keras.layers.Layer):
    def __init__(self, inplanes, ratio):
        super(Route, self).__init__()
        self.globalContext = GlobalContext(inplanes=inplanes, ratio=ratio)
        self.globalAvgPooling = keras.layers.GlobalAveragePooling2D()
        self.l2Norm = tf.math.l2_normalize
        self.dense = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = inputs
        x_ = self.globalContext(x)
        out = self.globalAvgPooling(x+x_)
        out = self.l2Norm(out)
        out = self.dense(out)
        return out


if __name__ == '__main__':
    # test GlobalContext
    img = tf.random.normal((2, 28, 28, 512))
    # gcLayer = GlobalContext(inplanes=1, ratio=2)
    # y = gcLayer(img)
    # print(y.shape)

    # test Route
    route = Route(inplanes=1, ratio=2)
    y = route(img)
    print(y.shape)