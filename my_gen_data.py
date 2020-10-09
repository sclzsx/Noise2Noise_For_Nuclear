import cv2
import random
import copy
from pathlib import Path
import numpy as np
import os

###########################################################################################################

# 从tiny数据集，删除黑边和底部时间戳，并使长宽可整除，生成图像对（干净，噪声，噪声，噪声，噪声，噪声）
def gen_newsize_img_pairs_of_tiny(path, shuffle=True):
    # 输入：图像，整除系数，左上角x，左上角y，右下角x，右下角y。输出：截取后的图
    def gen_newsize_image(input_image, scalar, tlx, tly, brx, bry):
        h, w, _ = input_image.shape
        new_h = h // scalar * scalar
        new_w = w // scalar * scalar
        image = input_image[tly:bry, tlx:brx]
        output_image = image[:new_h, :new_w]
        # print(output_image.shape[0], output_image.shape[1])
        # cv2.imshow('test', output_image)
        # cv2.waitKey()
        return output_image

    image_pairs = []
    dirs = [i for i in Path(path).iterdir() if i.is_dir()]
    for sub_dir in dirs:
        clean_img = cv2.imread(str(sub_dir) + '/0.png')
        noise_img1 = cv2.imread(str(sub_dir) + '/1.png')
        noise_img2 = cv2.imread(str(sub_dir) + '/2.png')
        noise_img3 = cv2.imread(str(sub_dir) + '/3.png')
        noise_img4 = cv2.imread(str(sub_dir) + '/4.png')

        new_clean_img = gen_newsize_image(clean_img, 16, 10, 80, 560, 420)
        new_noise_img1 = gen_newsize_image(noise_img1, 16, 10, 80, 560, 420)
        new_noise_img2 = gen_newsize_image(noise_img2, 16, 10, 80, 560, 420)
        new_noise_img3 = gen_newsize_image(noise_img3, 16, 10, 80, 560, 420)
        new_noise_img4 = gen_newsize_image(noise_img4, 16, 10, 80, 560, 420)

        pair_tmp = [new_clean_img, new_noise_img1, new_noise_img2, new_noise_img3, new_noise_img4]

        image_pairs.append(pair_tmp)

    if shuffle:
        random.shuffle(image_pairs)
    # print(len(image_pairs))
    return image_pairs

# 从图像对中划分出训练集， 验证集，测试集，比例6：2：2（保证训练集所占的比例只能大不能小）
def divide_img_pairs_into_3sets_of_tiny(image_pairs):
    pairs_num = len(image_pairs)
    val_num = int(pairs_num*0.2-1)
    val_img_pairs = image_pairs[:val_num + 1]
    test_img_pairs = image_pairs[val_num + 1:val_num + 1 + val_num + 1]
    train_img_pairs = image_pairs[val_num + 1 + val_num + 1:]
    # print(len(val_img_pairs), len(test_img_pairs), len(train_img_pairs))
    return train_img_pairs, val_img_pairs, test_img_pairs

# 对来自tiny的图像对，通过随机裁剪，生成图块对
def gen_random_patch_pairs_of_tiny(image_pairs, patch_size=64, patch_pre_img=16, shuffle=True):
    # 输入：图0，图1，图2，图3，图4. 输出：对每张输入图随机裁出同一位置大小的图像块
    def gen_random_patch(clean_img, noise_img1, noise_img2, noise_img3, noise_img4, patch_size):
        h, w, _ = clean_img.shape
        i = np.random.randint(h - patch_size + 1)
        j = np.random.randint(w - patch_size + 1)

        clean_patch = clean_img[i:i + patch_size, j:j + patch_size]
        noise_patch1 = noise_img1[i:i + patch_size, j:j + patch_size]
        noise_patch2 = noise_img2[i:i + patch_size, j:j + patch_size]
        noise_patch3 = noise_img3[i:i + patch_size, j:j + patch_size]
        noise_patch4 = noise_img4[i:i + patch_size, j:j + patch_size]
        a_random_patch_pair = [clean_patch, noise_patch1, noise_patch2, noise_patch3, noise_patch4]
        return a_random_patch_pair

    patch_pairs = []
    for pair in image_pairs:
        for i in range(patch_pre_img):
            pair_tmp = gen_random_patch(pair[0], pair[1], pair[2], pair[3], pair[4], patch_size)
            patch_pairs.append(pair_tmp)

    if shuffle:
        random.shuffle(patch_pairs)

    return patch_pairs

# 从tiny数据集的图像对，生成n2n训练集（“噪声-噪声”图块对）
def gen_n2n_trainset_of_tiny(patch_pairs):
    patch_pairs_num = len(patch_pairs)
    data_len = patch_pairs_num * 12
    # print(data_len)
    patch_size = patch_pairs[0][0].shape[0]
    train_data = np.zeros((data_len, patch_size, patch_size, 3), dtype=np.uint8)
    train_label = np.zeros((data_len, patch_size, patch_size, 3), dtype=np.uint8)
    pair_idx = 0
    for pair in patch_pairs:
        train_data[pair_idx] = pair[1]
        train_label[pair_idx] = pair[2]
        pair_idx += 1
        train_data[pair_idx] = pair[1]
        train_label[pair_idx] = pair[3]
        pair_idx += 1
        train_data[pair_idx] = pair[1]
        train_label[pair_idx] = pair[4]
        pair_idx += 1
        train_data[pair_idx] = pair[2]
        train_label[pair_idx] = pair[3]
        pair_idx += 1
        train_data[pair_idx] = pair[2]
        train_label[pair_idx] = pair[4]
        pair_idx += 1
        train_data[pair_idx] = pair[3]
        train_label[pair_idx] = pair[4]
        pair_idx += 1

        train_data[pair_idx] = pair[2]
        train_label[pair_idx] = pair[1]
        pair_idx += 1
        train_data[pair_idx] = pair[3]
        train_label[pair_idx] = pair[1]
        pair_idx += 1
        train_data[pair_idx] = pair[4]
        train_label[pair_idx] = pair[1]
        pair_idx += 1
        train_data[pair_idx] = pair[3]
        train_label[pair_idx] = pair[2]
        pair_idx += 1
        train_data[pair_idx] = pair[4]
        train_label[pair_idx] = pair[2]
        pair_idx += 1
        train_data[pair_idx] = pair[4]
        train_label[pair_idx] = pair[3]
        pair_idx += 1

    return train_data, train_label

# 从tiny数据集，生成n2c数据集（“噪声-干净”图像对或图块对）
def gen_n2c_dataset_of_tiny(patch_or_image_pairs):
    patch_pairs_num = len(patch_or_image_pairs)
    data_len = patch_pairs_num * 4
    # print(data_len)
    h = patch_or_image_pairs[0][0].shape[0]
    w = patch_or_image_pairs[0][0].shape[1]
    data = np.zeros((data_len, h, w, 3), dtype=np.uint8)
    label = np.zeros((data_len, h, w, 3), dtype=np.uint8)
    pair_idx = 0
    for pair in patch_or_image_pairs:
        data[pair_idx] = pair[1]
        label[pair_idx] = pair[0]
        pair_idx += 1
        data[pair_idx] = pair[2]
        label[pair_idx] = pair[0]
        pair_idx += 1
        data[pair_idx] = pair[3]
        label[pair_idx] = pair[0]
        pair_idx += 1
        data[pair_idx] = pair[4]
        label[pair_idx] = pair[0]
        pair_idx += 1

    return data, label

###########################################################################################################

def gen_pairs_from_full_random(path, patch_size, batch_size):
    def gen_random_patch(clean_img, noise_img1, noise_img2, patch_size):
        h, w, _ = clean_img.shape
        i = np.random.randint(h - patch_size + 1)
        j = np.random.randint(w - patch_size + 1)

        clean_patch = clean_img[i:i + patch_size, j:j + patch_size]
        noise_patch1 = noise_img1[i:i + patch_size, j:j + patch_size]
        noise_patch2 = noise_img2[i:i + patch_size, j:j + patch_size]

        tmp_pair = [clean_patch, noise_patch1, noise_patch2]
        return tmp_pair

    patch_pairs = []
    dirs = [i for i in Path(path).iterdir() if i.is_dir()]
    for sub_dir in dirs:
        clean_img = cv2.imread(str(sub_dir) + '/0.png')
        noise_img1 = cv2.imread(str(sub_dir) + '/1.png')
        noise_img2 = cv2.imread(str(sub_dir) + '/2.png')

        pair1 = gen_random_patch(clean_img, noise_img1, noise_img2, patch_size)
        pair2 = gen_random_patch(clean_img, noise_img1, noise_img2, patch_size)
        pair3 = gen_random_patch(clean_img, noise_img1, noise_img2, patch_size)
        pair4 = gen_random_patch(clean_img, noise_img1, noise_img2, patch_size)
        pair5 = gen_random_patch(clean_img, noise_img1, noise_img2, patch_size)
        pair6 = gen_random_patch(clean_img, noise_img1, noise_img2, patch_size)
        pair7 = gen_random_patch(clean_img, noise_img1, noise_img2, patch_size)
        pair8 = gen_random_patch(clean_img, noise_img1, noise_img2, patch_size)

        patch_pairs.append(pair1)
        patch_pairs.append(pair2)
        patch_pairs.append(pair3)
        patch_pairs.append(pair4)
        patch_pairs.append(pair5)
        patch_pairs.append(pair6)
        patch_pairs.append(pair7)
        patch_pairs.append(pair8)

    random.shuffle(patch_pairs)
    total_pair_num = len(patch_pairs)
    total_pair_num = total_pair_num // batch_size * batch_size
    used_patch_pairs = patch_pairs[:total_pair_num]

    return used_patch_pairs

def gen_train_from_full_random(train_path='', tr_patch_size=128, batch_size=8):
    train_pairs = gen_pairs_from_full_random(train_path, patch_size=tr_patch_size, batch_size=batch_size)
    print('train pair num:', len(train_pairs))
    train_data = np.zeros((len(train_pairs), tr_patch_size, tr_patch_size, 3), dtype=np.uint8)
    train_label = np.zeros((len(train_pairs), tr_patch_size, tr_patch_size, 3), dtype=np.uint8)
    pair_idx = 0
    for pair in train_pairs:
        train_data[pair_idx] = pair[1]
        train_label[pair_idx] = pair[2]
        pair_idx += 1
    return train_data, train_label

def gen_val_from_full_random(val_path='', val_patch_size=128, batch_size=8):
    val_patch_size = 64
    val_pairs = gen_pairs_from_full_random(val_path, patch_size=val_patch_size, batch_size=batch_size)
    print('val pair num (*2):', len(val_pairs) * 2)
    val_data = np.zeros((len(val_pairs) * 2, val_patch_size, val_patch_size, 3), dtype=np.uint8)
    val_label = np.zeros((len(val_pairs) * 2, val_patch_size, val_patch_size, 3), dtype=np.uint8)
    val_pair_idx = 0
    for pair in val_pairs:
        val_data[val_pair_idx] = pair[1]
        val_label[val_pair_idx] = pair[0]
        val_pair_idx += 1
        val_data[val_pair_idx] = pair[2]
        val_label[val_pair_idx] = pair[0]
        val_pair_idx += 1
    return val_data, val_label

def gen_pairs_from_full_sliding(path, patch_size, stride, batch_size):
    patch_pairs = []
    dirs = [i for i in Path(path).iterdir() if i.is_dir()]
    for sub_dir in dirs:
        clean_img = cv2.imread(str(sub_dir) + '/0.png')
        noise_img1 = cv2.imread(str(sub_dir) + '/1.png')
        noise_img2 = cv2.imread(str(sub_dir) + '/2.png')
        h, w, _ = clean_img.shape
        for x in range(0, h - patch_size + 1, stride):
            for y in range(0, w - patch_size + 1, stride):
                clean_img_clone = copy.deepcopy(clean_img)
                noise_img1_clone = copy.deepcopy(noise_img1)
                noise_img2_clone = copy.deepcopy(noise_img2)

                clean_patch = clean_img_clone[x: x + patch_size, y: y + patch_size]
                noise_patch1 = noise_img1_clone[x: x + patch_size, y: y + patch_size]
                noise_patch2 = noise_img2_clone[x: x + patch_size, y: y + patch_size]

                # cv2.imshow("c", clean_patch)
                # cv2.imshow("1", noise_patch1)
                # cv2.imshow("2", noise_patch2)
                # cv2.waitKey()
                # cv2.rectangle(clean_img, (y, x), (y + patch_size, x + patch_size), (255, 0, 0), 2)

                tmp_pair = [clean_patch, noise_patch1, noise_patch2]

                patch_pairs.append(tmp_pair)
        # cv2.imshow("ss", clean_img)
        # cv2.waitKey()

    random.shuffle(patch_pairs)
    total_pair_num = len(patch_pairs)
    total_pair_num = total_pair_num // batch_size * batch_size
    used_patch_pairs = patch_pairs[:total_pair_num]

    return used_patch_pairs

###########################################################################################################

def gen_pairs_from_src37(data_dir, batch_size=128, scene_num=37, stride=20, patch_size=40):
    patch_pairs = []

    for clean_idx in range(1, scene_num + 1):
        clean_img_name = str(data_dir + '/' + str(clean_idx) + '.png')
        # print(clean_img_name)
        clean_img = cv2.imread(clean_img_name)

        noise_dir = data_dir + '/' + str(clean_idx)
        noise_images_this_dir = [i for i in Path(noise_dir).iterdir() if i.is_file()]

        noise_img1_name = random.choice(noise_images_this_dir)
        noise_img1 = cv2.imread(str(noise_img1_name))
        noise_images_this_dir.remove(noise_img1_name)

        noise_img2_name = random.choice(noise_images_this_dir)
        noise_img2 = cv2.imread(str(noise_img2_name))
        noise_images_this_dir.remove(noise_img2_name)

        noise_img3_name = random.choice(noise_images_this_dir)
        noise_img3 = cv2.imread(str(noise_img3_name))

        h, w, _ = clean_img.shape
        # scales = [1, 0.9, 0.8, 0.7]
        scales = [1]
        for s in scales:
            h_scaled, w_scaled = int(h * s), int(w * s)

            clean_img_scaled = cv2.resize(clean_img, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
            noise_img_scaled1 = cv2.resize(noise_img1, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
            noise_img_scaled2 = cv2.resize(noise_img2, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
            noise_img_scaled3 = cv2.resize(noise_img3, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
            noise_images = [noise_img_scaled1, noise_img_scaled2, noise_img_scaled3]

            for noise_img_scaled in noise_images:
                clean_img_scaled_clone = copy.deepcopy(clean_img_scaled)

                for i in range(20, h_scaled - patch_size - 40 + 1, stride):
                    for j in range(20, w_scaled - patch_size + 1, stride):
                        data = noise_img_scaled[i:i + patch_size, j:j + patch_size]
                        label = clean_img_scaled_clone[i:i + patch_size, j:j + patch_size]

                        a_patch_pair = [data, label]
                        patch_pairs.append(a_patch_pair)

    random.shuffle(patch_pairs)
    total_pair_num = len(patch_pairs)
    total_pair_num = total_pair_num // batch_size * batch_size
    used_patch_pairs = patch_pairs[:total_pair_num]

    train_pair_num = int(total_pair_num * 0.7)
    # val_pair_num = int(total_pair_num * 0.3)
    # print(total_pair_num, train_pair_num, val_pair_num)

    steps_a_epoch = train_pair_num // batch_size

    used_train_pair_num = batch_size * steps_a_epoch
    patch_pairs_train = patch_pairs[:used_train_pair_num]

    del patch_pairs[:used_train_pair_num]
    used_val_pair_num = len(patch_pairs) // batch_size * batch_size
    patch_pairs_val = patch_pairs[:used_val_pair_num]

    return steps_a_epoch, patch_pairs_train, patch_pairs_val, used_patch_pairs

def gen_images_from_src37(src_path='../37/test', dst_path='../sx_noise_data_tiny/test', create_tiny=1):
    pair_idx = 0
    tr_cleans = [i for i in Path(src_path).iterdir() if i.is_file()]
    for clean_name in tr_cleans:
        clean_name = str(clean_name)
        clean = cv2.imread(clean_name)
        if clean is None:
            continue

        correspond_noise_dir = clean_name[:-4]
        # print(noise_dir_name)
        noises_this_dir = [i for i in Path(correspond_noise_dir).iterdir() if i.is_file()]
        random.shuffle(noises_this_dir)

        if create_tiny:
            noise_name1 = noises_this_dir[0]
            noise_name2 = noises_this_dir[1]
            noise_name3 = noises_this_dir[2]
            noise_name4 = noises_this_dir[3]
            noise1 = cv2.imread(str(noise_name1))
            noise2 = cv2.imread(str(noise_name2))
            noise3 = cv2.imread(str(noise_name3))
            noise4 = cv2.imread(str(noise_name4))
            if noise1 is None:
                continue
            if noise2 is None:
                continue
            if noise3 is None:
                continue
            if noise4 is None:
                continue
            # cv2.imshow('0', clean)
            # cv2.imshow('1', noise1)
            # cv2.imshow('2', noise2)
            # cv2.waitKey()

            save_dir = dst_path + '/' + str(pair_idx)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(save_dir + '/0.png', clean)
            cv2.imwrite(save_dir + '/1.png', noise1)
            cv2.imwrite(save_dir + '/2.png', noise2)
            cv2.imwrite(save_dir + '/3.png', noise3)
            cv2.imwrite(save_dir + '/4.png', noise4)

            print('processed pair idx: ', pair_idx)
            pair_idx += 1

        else:
            for i, noise_name in enumerate(noises_this_dir):
                if i == len(noises_this_dir) - 1:
                    break
                noise_name1 = noises_this_dir[i]
                noise_name2 = noises_this_dir[i + 1]

                noise1 = cv2.imread(str(noise_name1))
                noise2 = cv2.imread(str(noise_name2))
                if noise1 is None:
                    continue
                if noise2 is None:
                    continue

                # cv2.imshow('0', clean)
                # cv2.imshow('1', noise1)
                # cv2.imshow('2', noise2)
                # cv2.waitKey()

                save_dir = dst_path + '/' + str(pair_idx)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                cv2.imwrite(save_dir + '/0.png', clean)
                cv2.imwrite(save_dir + '/1.png', noise1)
                cv2.imwrite(save_dir + '/2.png', noise2)

                print('processed pair idx: ', pair_idx)
                pair_idx += 1

###########################################################################################################

if __name__ == '__main__':
    print('test information while coding')

    # image_pairs = gen_newsize_img_pairs_of_tiny('data_tiny', shuffle=True)
    # train_image_pairs, val_image_pairs, test_image_pairs = divide_img_pairs_into_3sets_of_tiny(image_pairs)
    # train_patch_pairs_n2n = gen_random_patch_pairs_of_tiny(train_image_pairs, 64, 16, shuffle=True)
    # train_patch_pairs_n2c = gen_random_patch_pairs_of_tiny(train_image_pairs, 64, 48, shuffle=True)
    #
    # train_data_n2n, train_label_n2n = gen_n2n_trainset_of_tiny(train_patch_pairs_n2n)
    # train_data_n2c, train_label_n2c = gen_n2c_dataset_of_tiny(train_patch_pairs_n2c)
    #
    # val_data, val_label = gen_n2c_dataset_of_tiny(val_image_pairs)
    # test_data, test_label = gen_n2c_dataset_of_tiny(test_image_pairs)

    # print(len(val_data), len(test_data))

