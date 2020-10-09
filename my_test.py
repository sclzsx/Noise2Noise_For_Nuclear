import os
from pathlib import Path
import numpy as np
import math
import time
import cv2
import random

import my_models

import tensorflow as tf

import glob


def find_latest_epoch_h5(dir):
    file_paths = glob.glob(os.path.join(dir, '*.hdf5'))
    latest_epoch_path = ''
    latest_epoch_name = ''
    latest_epoch = 0

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        epoch_num_str = file_name[:3]
        epoch_num = int(epoch_num_str[0]) * 100 + int(epoch_num_str[1]) * 10 + int(epoch_num_str[2])
        if latest_epoch <= epoch_num:
            latest_epoch = epoch_num
            latest_epoch_path = file_path
            latest_epoch_name = file_name

    return latest_epoch_path, latest_epoch_name
    # print(latest_epoch_path)


def my_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test_path = './data_tiny/val'

    write_flag = 0
    show_flag = 0

    results_dir = '../trained_results-12-28'
    results_sub_dir_name = 'dncnn_l0_ps64_bs4_ppi48'
    weights_dir = results_dir + '/' + results_sub_dir_name
    latest_epoch_path, latest_epoch_name = find_latest_epoch_h5(weights_dir)

    model_name_char_list = []
    for char_idx in range(len(results_sub_dir_name)):
        char = results_sub_dir_name[char_idx]
        model_name_char_list.append(char)
        if latest_epoch_name[char_idx - 1] == '-':
            break
    model_name = ''.join(model_name_char_list)
    # print(model_name)

    model = my_models.get_model(model_name)

    model.load_weights(latest_epoch_path)

    def psnr(img1, img2):
        mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
        if mse < 1.0e-10:
            return 100
        return 10 * math.log10(255.0 ** 2 / mse)

    def get_test_info_n2n(clean_img, noise_img):
        h, w, _ = clean_img.shape
        clean_img = clean_img[:(h // 16) * 16, :(w // 16) * 16]
        noise_img = noise_img[:(h // 16) * 16, :(w // 16) * 16]
        h2, w2, _ = clean_img.shape

        start = time.time()
        pred = model.predict(np.expand_dims(noise_img, 0))
        denoised = np.clip(pred[0], 0, 255).astype(dtype=np.uint8)
        end = time.time()
        runtime = end - start

        psnr_before = psnr(clean_img, noise_img)
        psnr_after = psnr(clean_img, denoised)

        out_image = np.zeros((h2, w2 * 3, 3), dtype=np.uint8)
        out_image[:, :w] = clean_img
        out_image[:, w:w * 2] = noise_img
        out_image[:, w * 2:] = denoised

        return runtime, psnr_before, psnr_after, out_image

    txt_path = str(weights_dir) + '/cal.txt'
    # print(txt_path)
    with open(txt_path, 'w') as file_object:
        avg_psnr_before = []
        avg_psnr_after = []
        avg_runtime = []
        test_count = 1

        dirs = [i for i in Path(test_path).iterdir() if i.is_dir()]
        random.shuffle(dirs)
        for sub_dir in dirs:
            clean_img = cv2.imread(str(sub_dir) + '/0.png')
            noise_img1 = cv2.imread(str(sub_dir) + '/1.png')
            noise_img2 = cv2.imread(str(sub_dir) + '/2.png')

            runtime1, psnr_before1, psnr_after1, out_image1 = get_test_info_n2n(clean_img, noise_img1)
            runtime2, psnr_before2, psnr_after2, out_image2 = get_test_info_n2n(clean_img, noise_img2)

            if write_flag:
                cv2.imwrite(str(weights_dir) + '/' + str(test_count) + ".png", out_image1)
                cv2.imwrite(str(weights_dir) + '/' + str(test_count) + ".png", out_image2)

            if show_flag:
                h3, w3, _ = out_image1.shape
                show_out1 = cv2.resize(out_image1, (w3 // 16 * 8, h3 // 16 * 8), interpolation=cv2.INTER_CUBIC)
                show_out2 = cv2.resize(out_image2, (w3 // 16 * 8, h3 // 16 * 8), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("o1", show_out1)
                cv2.imshow("o2", show_out2)
                cv2.waitKey()

            write_info1 = 'idx: ' + str(test_count) + '\ttime: ' + str(runtime1) + '\tpsnr_before: ' + \
                          str(psnr_before1) + '\tpsnr_after: ' + str(psnr_after1)
            print(write_info1)

            write_info2 = 'idx: ' + str(test_count + 1) + '\ttime: ' + str(runtime2) + '\tpsnr_before: ' + \
                          str(psnr_before2) + '\tpsnr_after: ' + str(psnr_after2)
            print(write_info2)

            file_object.write(write_info1 + '\n')
            file_object.write(write_info2 + '\n')

            test_count += 2

            avg_runtime.append(runtime1)
            avg_psnr_before.append(psnr_before1)
            avg_psnr_after.append(psnr_after1)

            avg_runtime.append(runtime2)
            avg_psnr_before.append(psnr_before2)
            avg_psnr_after.append(psnr_after2)

        test_num = len(avg_runtime)
        avg_runtime = float(sum(avg_runtime)) / test_num
        avg_psnr_before = float(sum(avg_psnr_before)) / test_num
        avg_psnr_after = float(sum(avg_psnr_after)) / test_num

        info = '\n\navg denoising time: ' + str(avg_runtime) + '\tavg_psnr_before: ' + str(
            avg_psnr_before) + '\tavg_psnr_after: ' + str(avg_psnr_after)
        file_object.write(info + '\n')
        print(info)


if __name__ == '__main__':
    my_test()
