import my_gen_data
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import os
from keras.models import Model
from keras.layers import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


image_pairs = my_gen_data.gen_newsize_img_pairs_of_tiny('data_tiny', shuffle=True)
train_image_pairs, val_image_pairs, test_image_pairs = my_gen_data.divide_img_pairs_into_3sets_of_tiny(image_pairs)

train_patch_pairs_n2c = my_gen_data.gen_random_patch_pairs_of_tiny(train_image_pairs, 64, 10, shuffle=True)

train_data_n2c, train_label_n2c = my_gen_data.gen_n2c_dataset_of_tiny(train_patch_pairs_n2c)
val_data, val_label = my_gen_data.gen_n2c_dataset_of_tiny(val_image_pairs)
test_data, test_label = my_gen_data.gen_n2c_dataset_of_tiny(test_image_pairs)

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

def train_model(save_dir, model_name, loss_type):
    model = dncnn()

    output_path = save_dir + '/' + model_name + '-' + loss_type
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    callbacks = []

    model.compile(optimizer=Adam(lr=1e-4), loss=loss_type, metrics=[PSNR])

    callbacks.append(ModelCheckpoint(str(output_path) + "/{epoch:03d}-{val_loss:.3f}-{val_PSNR:.3f}.hdf5",
                                     monitor="val_PSNR", verbose=1, mode="max", save_best_only=True))

    tr_data = train_data_n2c
    tr_label = train_label_n2c

    hist = model.fit(tr_data, tr_label, epochs=50, batch_size=4, verbose=1,
                     callbacks=callbacks, validation_data=(val_data, val_label), shuffle=True)
    np.savez(str(output_path + '/history.npz'), history=hist.history)

def denoise_a_image(weight_filepath, noise_img):
    model = dncnn()
    model.load_weights(weight_filepath)
    pred = model.predict(np.expand_dims(noise_img, 0))
    denoised = np.clip(pred[0], 0, 255).astype(dtype=np.uint8)
    return denoised

if __name__ == '__main__':

    save_root = 'exp-1-13/'
    dirname = 'nuclear_tiny_dncnn'
    model_name = 'dncnn'
    loss_type = 'mse'
    train_model(save_root + dirname, model_name, loss_type)

    # denoised = denoise_a_image('exp-1-13/nuclear_tiny_dncnn/dncnn-mse/example.hdf5', test_data[0])
    # cv2.imshow('denoised', denoised)
    # cv2.waitKey()
