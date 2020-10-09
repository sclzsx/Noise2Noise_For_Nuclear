import keras
from keras.models import Model
from keras.layers import *

from batch_renorm import BatchRenormalization

def get_model(model_name="srresnet"):
    if model_name == "srresnet":
        return srresnet()
    elif model_name == "unet":
        return unet(out_ch=3)
    elif model_name == "dncnn":
        return dncnn()
    elif model_name == "brdnet":
        return brdnet()

    elif model_name == "ours_brn":
        return ours_brn()
    elif model_name == "ours_bn":
        return ours_bn()
    elif model_name == "ours_no_batch":
        return ours_no_batch()

def srresnet(input_channel_num=3, feature_dim=64, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])
        return m

    inputs = Input(shape=(None, None, input_channel_num))
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model

def unet(input_channel_num=3, out_ch=3, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    def _conv_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same')(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, activation=acti, padding='same')(n)
        n = BatchNormalization()(n) if bn else n
        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, res)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, acti, bn, res)
        else:
            m = _conv_block(m, dim, acti, bn, res, do)
        return m

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1)(o)
    model = Model(inputs=i, outputs=o)

    return model

def brdnet(): #original format def BRDNet(), data is used to obtain the reshape of input data
    inpt = Input(shape=(None,None,3))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = BatchRenormalization(axis=-1, epsilon=1e-3)(x)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(7):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchRenormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    # last layer, Conv
    for i in range(8):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchRenormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(x) #gray is 1 color is 3
    x = keras.layers.Subtract()([inpt, x])   # input - noise
    y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    y = BatchRenormalization(axis=-1, epsilon=1e-3)(y)
    y = Activation('relu')(y)
    # 15 layers, Conv+BN+relu
    for i in range(7):
        y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),dilation_rate=(2,2), padding='same')(y)
        y = Activation('relu')(y)
    y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(y)
    y = BatchRenormalization(axis=-1, epsilon=1e-3)(y)
    y = Activation('relu')(y)
    for i in range(6):
        y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),dilation_rate=(2,2), padding='same')(y)
        y = Activation('relu')(y)
    y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(y)
    y = BatchRenormalization(axis=-1, epsilon=1e-3)(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(y)#gray is 1 color is 3
    y = keras.layers.Subtract()([inpt, y])   # input - noise

    o = concatenate([x, y], axis=-1)
    z = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(o)#gray is 1 color is 3
    z = keras.layers.Subtract()([inpt, z])
    model = Model(inputs=inpt, outputs=z)
    return model

def dncnn(depth=20, filters=64, image_channels=3, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth - 2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn' + str(layer_count))(x)
        layer_count += 1
        x = Activation('relu', name='relu' + str(layer_count))(x)
        # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same', use_bias=False, name='conv' + str(layer_count))(x)
    layer_count += 1
    x = Subtract(name='subtract' + str(layer_count))([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x)

    return model

def ours_bn(input_channel_num=3, feature_dim=64, resunit_num=16):
    inputs = Input(shape=(None, None, input_channel_num))

    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])
        return m

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x
    for i in range(resunit_num):
        x = _residual_block(x)
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])

    y = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    y = PReLU(shared_axes=[1, 2])(y)
    y0 = y
    for i in range(resunit_num):
        y = _residual_block(y)
    y = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(y)
    y = BatchNormalization()(y)
    y = Add()([y, y0])

    o = Concatenate(axis=-1)([x, y])
    # o = K.concatenate([x, y], axis=-1)
    outputs = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(o)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def ours_no_batch(input_channel_num=3, feature_dim=64, resunit_num=16):
    inputs = Input(shape=(None, None, input_channel_num))

    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        # x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        # x = BatchNormalization()(x)
        m = Add()([x, inputs])
        return m

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x
    for i in range(resunit_num):
        x = _residual_block(x)
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    # x = BatchNormalization()(x)
    x = Add()([x, x0])

    y = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    y = PReLU(shared_axes=[1, 2])(y)
    y0 = y
    for i in range(resunit_num):
        y = _residual_block(y)
    y = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(y)
    # y = BatchNormalization()(y)
    y = Add()([y, y0])

    o = Concatenate(axis=-1)([x, y])
    # o = K.concatenate([x, y], axis=-1)
    outputs = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(o)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def ours_brn(input_channel_num=3, feature_dim=64, resunit_num=16):
    inputs = Input(shape=(None, None, input_channel_num))

    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchRenormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchRenormalization()(x)
        m = Add()([x, inputs])
        return m

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x
    for i in range(resunit_num):
        x = _residual_block(x)
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchRenormalization()(x)
    x = Add()([x, x0])

    y = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    y = PReLU(shared_axes=[1, 2])(y)
    y0 = y
    for i in range(resunit_num):
        y = _residual_block(y)
    y = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(y)
    y = BatchRenormalization()(y)
    y = Add()([y, y0])

    o = Concatenate(axis=-1)([x, y])
    # o = K.concatenate([x, y], axis=-1)
    outputs = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(o)
    model = Model(inputs=inputs, outputs=outputs)

    return model

