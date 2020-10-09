import my_gen_data
import my_models
import my_utils

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

image_pairs = my_gen_data.gen_newsize_img_pairs_of_tiny('data_tiny', shuffle=True)
train_image_pairs, val_image_pairs, test_image_pairs = my_gen_data.divide_img_pairs_into_3sets_of_tiny(image_pairs)
train_patch_pairs_n2n = my_gen_data.gen_random_patch_pairs_of_tiny(train_image_pairs, 64, 21, shuffle=True)

# 把测试集划分给n2c训练
train_patch_pairs_n2c = my_gen_data.gen_random_patch_pairs_of_tiny(test_image_pairs, 64, 207, shuffle=True)

train_data_n2n, train_label_n2n = my_gen_data.gen_n2n_trainset_of_tiny(train_patch_pairs_n2n)
train_data_n2c, train_label_n2c = my_gen_data.gen_n2c_dataset_of_tiny(train_patch_pairs_n2c)
val_data, val_label = my_gen_data.gen_n2c_dataset_of_tiny(val_image_pairs)
# test_data, test_label = my_gen_data.gen_n2c_dataset_of_tiny(test_image_pairs)

def train_model(save_dir, model_name, loss_type, train_n2c=False):
    model = my_models.get_model(model_name)

    output_path = save_dir + '/' + model_name + '-' + loss_type
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    callbacks = []

    if loss_type == "l0":
        l0 = my_utils.L0Loss()
        callbacks.append(my_utils.UpdateAnnealingParameter(l0.gamma, 50, verbose=1))
        loss_type = l0()

    model.compile(optimizer=Adam(lr=1e-4), loss=loss_type, metrics=[my_utils.PSNR])

    callbacks.append(ModelCheckpoint(str(output_path) + "/{epoch:03d}-{val_loss:.3f}-{val_PSNR:.3f}.hdf5",
                                     monitor="val_PSNR", verbose=1, mode="max", save_best_only=True))

    tr_data = train_data_n2n
    tr_label = train_label_n2n
    if train_n2c:
        tr_data = train_data_n2c
        tr_label = train_label_n2c

    hist = model.fit(tr_data, tr_label, epochs=50, batch_size=4, verbose=1,
                     callbacks=callbacks, validation_data=(val_data, val_label), shuffle=True)
    np.savez(str(output_path + '/history.npz'), history=hist.history)


def exp_figure2(save_root):
    dirname = 'figure2'
    train_model(save_root + dirname, 'ours_no_batch', 'l0')
    train_model(save_root + dirname, 'ours_bn', 'l0')
    train_model(save_root + dirname, 'ours_brn', 'l0')

# 注释掉表示前面实验已有，无需再做
def exp_table1(save_root):
    dirname = 'table1'
    train_model(save_root + dirname, 'srresnet', 'l0')
    # train_model(save_root + dirname, 'unet', 'l0')
    train_model(save_root + dirname, 'ours_brn', 'l0')

    train_model(save_root + dirname, 'srresnet', 'mae')
    # train_model(save_root + dirname, 'unet', 'mae')
    train_model(save_root + dirname, 'ours_brn', 'mae')

    train_model(save_root + dirname, 'srresnet', 'mse')
    # train_model(save_root + dirname, 'unet', 'mse')
    train_model(save_root + dirname, 'ours_brn', 'mse')

# 注释掉表示前面实验已有，无需再做
def exp_table2(save_root):
    dirname = 'table2'
    train_model(save_root + dirname, 'dncnn', 'l0', train_n2c=True)
    train_model(save_root + dirname, 'brdnet', 'l0', train_n2c=True)
    # train_model(save_root + dirname, 'unet', 'l0')
    # train_model(save_root + dirname, 'srresnet', 'l0')
    # train_model(save_root + dirname, 'ours_brn', 'l0')


if __name__ == '__main__':
    save_root = 'exp-1-1/'

    # exp_figure2(save_root)

    exp_table1(save_root)

    exp_table2(save_root)

