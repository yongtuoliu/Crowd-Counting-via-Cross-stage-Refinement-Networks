import os.path as ops
import h5py
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as CM


class DataSet(object):
    def __init__(self, dataset_info_file):
        """
        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_list = self._init_dataset(dataset_info_file)
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):
        """
        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_label_list = []

        assert ops.exists(dataset_info_file), '{:s}　not exists'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                gt_img_list.append(info_tmp[1])
                gt_label_list.append(info_tmp[0])
        return gt_img_list, gt_label_list

    def _generate_training_pathches(self, gt_img, patch_nums):
        """
        :param gt_img:
        :param label_img:
        :param patch_nums:
        :param patch_size:
        :return:
        """
        height = gt_img.shape[0]
        weight = gt_img.shape[1]
        
        gt_img_patches = []
        gt_img_patch_1 = gt_img[0:height//2, 0:weight//2, :]
        gt_img_patch_2 = gt_img[0:height//2, weight//4:(weight//2+weight//4), :]
        gt_img_patch_3 = gt_img[0:height//2, weight//2:weight, :]
        gt_img_patch_4 = gt_img[height//4:(height//2+height//4), 0:weight//2, :]
        gt_img_patch_5 = gt_img[height//4:(height//2+height//4), weight//4:(weight//2+weight//4), :]
        gt_img_patch_6 = gt_img[height//4:(height//2+height//4), weight//2:weight, :]
        gt_img_patch_7 = gt_img[height//2:height, 0:weight//2, :]
        gt_img_patch_8 = gt_img[height//2:height, weight//4:(weight//2+weight//4), :]
        gt_img_patch_9 = gt_img[height//2:height, weight//2:weight, :]
        gt_img_patches.append(gt_img_patch_1)
        gt_img_patches.append(gt_img_patch_2)
        gt_img_patches.append(gt_img_patch_3)
        gt_img_patches.append(gt_img_patch_4)
        gt_img_patches.append(gt_img_patch_5)
        gt_img_patches.append(gt_img_patch_6)
        gt_img_patches.append(gt_img_patch_7)
        gt_img_patches.append(gt_img_patch_8)
        gt_img_patches.append(gt_img_patch_9)
        
        # 1/2 and 1/4 and 1/8
        gt_img_patches_2 = []
        gt_img_patches_4 = []
        gt_img_patches_8 = []
        gt_img_patch_2_1 = cv2.resize(gt_img_patch_1, (int(gt_img_patch_1.shape[1]/2), int(gt_img_patch_1.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
        gt_img_patch_4_1 = cv2.resize(gt_img_patch_1, (int(gt_img_patch_1.shape[1]/4), int(gt_img_patch_1.shape[0]/4)), interpolation = cv2.INTER_CUBIC)
        gt_img_patch_8_1 = cv2.resize(gt_img_patch_1, (int(gt_img_patch_1.shape[1]/8), int(gt_img_patch_1.shape[0]/8)), interpolation = cv2.INTER_CUBIC)
        gt_img_patches_4.append(gt_img_patch_4_1)
        gt_img_patches_4.append(gt_img_patch_4_1)
        gt_img_patches_4.append(gt_img_patch_4_1)
        gt_img_patches_4.append(gt_img_patch_4_1)
        gt_img_patches_4.append(gt_img_patch_4_1)
        gt_img_patches_4.append(gt_img_patch_4_1)
        gt_img_patches_4.append(gt_img_patch_4_1)
        gt_img_patches_4.append(gt_img_patch_4_1)
        gt_img_patches_4.append(gt_img_patch_4_1)
        
        gt_img_patches_2.append(gt_img_patch_2_1)
        gt_img_patches_2.append(gt_img_patch_2_1)
        gt_img_patches_2.append(gt_img_patch_2_1)
        gt_img_patches_2.append(gt_img_patch_2_1)
        gt_img_patches_2.append(gt_img_patch_2_1)
        gt_img_patches_2.append(gt_img_patch_2_1)
        gt_img_patches_2.append(gt_img_patch_2_1)
        gt_img_patches_2.append(gt_img_patch_2_1)
        gt_img_patches_2.append(gt_img_patch_2_1)

        gt_img_patches_8.append(gt_img_patch_8_1)
        gt_img_patches_8.append(gt_img_patch_8_1)
        gt_img_patches_8.append(gt_img_patch_8_1)
        gt_img_patches_8.append(gt_img_patch_8_1)
        gt_img_patches_8.append(gt_img_patch_8_1)
        gt_img_patches_8.append(gt_img_patch_8_1)
        gt_img_patches_8.append(gt_img_patch_8_1)
        gt_img_patches_8.append(gt_img_patch_8_1)
        gt_img_patches_8.append(gt_img_patch_8_1)

        return gt_img_patches, gt_img_patches_2, gt_img_patches_4, gt_img_patches_8

    def next_batch(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_list):
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_list = self._gt_label_list[idx_start:idx_end]

            img_patches = []
            img_patches_2 = []
            img_patches_4 = []
            img_patches_8 = []
            gt_images = []
            gt_labels = []

            for index, gt_img_path in enumerate(gt_img_list):
                gt_image = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)

                label_file = h5py.File(gt_label_list[index])
                label_image = np.asarray(label_file['density'])
                label_image = np.expand_dims(label_image, axis=-1)

                height = gt_image.shape[0]
                weight = gt_image.shape[1]
                h_residual = height % 32
                w_residual = weight % 32
                if(h_residual!=0):
                    gt_image = gt_image[0:height-h_residual, :, :]
                    label_image = label_image[0:height-h_residual, :, :]
                if(w_residual!=0):
                    gt_image = gt_image[:, 0:weight-w_residual, :]
                    label_image = label_image[:, 0:weight-w_residual, :]

                height = gt_image.shape[0]
                weight = gt_image.shape[1]

                gt_images.append(gt_image)
                gt_labels.append(label_image)

                gt_image = np.divide(gt_image, 255)

                gt_image_patches, gt_image_patches_2, gt_image_patches_4, gt_image_patches_8 = self._generate_training_pathches(gt_img=gt_image, patch_nums=9)
                for index, gt_image_patch in enumerate(gt_image_patches):
                    img_patches.append(gt_image_patch)
                    img_patches_2.append(gt_image_patches_2[index])
                    img_patches_4.append(gt_image_patches_4[index])
                    img_patches_8.append(gt_image_patches_8[index])

            self._next_batch_loop_count += 1
            return img_patches, height, weight, img_patches_2, img_patches_4, img_patches_8, gt_labels, gt_images


if __name__ == '__main__':
    return