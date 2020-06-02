import os
import os.path as ops
import tensorflow as tf
import numpy as np
import utils
import sys
from counting_model import counting_net

import cv2

import matplotlib
matplotlib.use('Agg') #　not show up just write into disk
from matplotlib import pyplot as plt

import train_data_provider
import validate_data_provider

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

dataset_dir = "./ShanghaiTech_Crowd_Counting_Dataset/part_A_final/"
output_density_map = "./output"
batch_size = 1
loss_c_weight = 0.001
loss_weight = 0.5

if __name__ == "__main__":
    validate_dataset = validate_data_provider.DataSet(ops.join(dataset_dir, 'test.txt'))

    # declare tensor
    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    x_2 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_2")
    x_4 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_4")
    x_8 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_8")
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="label")
    phase_tensor = tf.placeholder(tf.string, shape=[], name='phase')


    # 声明网络，计算loss
    density_map_list_typelist = counting_net.CountingNet(phase_tensor).build(x, x_2, x_4, x_8, name='density_map_ret')
    density_map_list = tf.convert_to_tensor(density_map_list_typelist)
    estimated_counting_2 = tf.reduce_sum(density_map_list[2], reduction_indices=[1, 2, 3], name="crowd_counting_2")
    ground_truth_counting = tf.cast(tf.reduce_sum(y, reduction_indices=[1, 2, 3]), tf.float32)
    eval_metric_ops = {
        'MAE': tf.reduce_mean(tf.abs(tf.subtract(estimated_counting_2, ground_truth_counting)), axis=0, name="MAE"),
        'MSE': tf.reduce_mean(tf.square(tf.subtract(ground_truth_counting, estimated_counting_2)), axis=0, name="MSE"),
    }
    loss_e = tf.constant(0.0, tf.float32)
    loss_c = tf.constant(0.0, tf.float32)
    n = density_map_list.get_shape().as_list()[0]
    for index, density_map in enumerate(density_map_list_typelist):
        mse_loss = tf.multiply(loss_weight, tf.losses.mean_squared_error(labels=y, predictions=density_map_list[index]))
        ssim_loss = tf.multiply(loss_weight, utils.structural_similarity_index_metric(density_map_list[index], y))
        loss_e = tf.add(loss_e, mse_loss)
        loss_c = tf.add(loss_c, ssim_loss)
    loss = tf.add(loss_e, tf.multiply(loss_c_weight, loss_c))

    # for visulization, 3600 and 182 are consistent with the training and validating numbers of batches in each epoch
    z4 = tf.placeholder(tf.float32, shape=[182, 1], name="loss_validate")
    h4 = tf.reduce_mean(z4, name="loss_validate_mean")
    z5 = tf.placeholder(tf.float32, shape=[182], name="MAE_validate")
    h5 = tf.reduce_mean(z5, axis=0, name="MAE_validate_mean")
    z6 = tf.placeholder(tf.float32, shape=[182], name="MSE_validate")
    h6 = tf.sqrt(tf.reduce_mean(z6, axis=0), name="RMSE_validate")


    #　set tf saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # model restoration
        weights_path = './checkpoint_dir/counting_epoch256.ckpt'
        saver.restore(sess=sess, save_path=weights_path)     

        # validate
        MAE_validate_ = []
        MSE_validate_ = []
        image_validate_num = 182
        # output density maps
        density_map_dir = output_density_map + "/epoch" + str(i)
        if not ops.exists(density_map_dir):
            os.makedirs(density_map_dir)
        figure, axes = plt.subplots(1, 3, figsize=(20, 5))
        for k in range(image_validate_num):
            density_map_path = density_map_dir +"/d_map" + str(m + 1) + ".png"
            gt_imgs_9patches, height, weight, gt_imgs_9patches_2, gt_imgs_9patches_4, gt_imgs_9patches_8, label_imgs, gt_imgs = validate_dataset.next_batch(batch_size)
            density_map_list_9patches = sess.run(density_map_list_typelist, feed_dict={x: gt_imgs_9patches, x_2: gt_imgs_9patches_2, x_4: gt_imgs_9patches_4, x_8: gt_imgs_9patches_8, phase_tensor: 'test'})
            n = len(density_map_list_9patches) # how many steps
            density_map_full_list = np.zeros((n, batch_size, height, weight, 1))
            for l in range(n):
                density_map_full_list[l][0][0:(height//4+height//8), 0:(weight//4+weight//8), :] = density_map_list_9patches[l][0][0:(height//4+height//8), 0:(weight//4+weight//8), :]
                density_map_full_list[l][0][0:(height//4+height//8), (weight//4+weight//8):(weight//2+weight//8), :] = density_map_list_9patches[l][1][0:(height//4+height//8), weight//8:(weight//2-weight//4+weight//8), :]
                density_map_full_list[l][0][0:(height//4+height//8), (weight//2+weight//8):weight, :] = density_map_list_9patches[l][2][0:(height//4+height//8), weight//8:(weight-weight//2), :]

                density_map_full_list[l][0][(height//4+height//8):(height//2+height//8), 0:(weight//4+weight//8), :] = density_map_list_9patches[l][3][height//8:(height//2-height//4+height//8), 0:(weight//4+weight//8), :]
                density_map_full_list[l][0][(height//4+height//8):(height//2+height//8), (weight//4+weight//8):(weight//2+weight//8), :] = density_map_list_9patches[l][4][height//8:(height//2-height//4+height//8), weight//8:(weight//2-weight//4+weight//8), :]
                density_map_full_list[l][0][(height//4+height//8):(height//2+height//8), (weight//2+weight//8):weight, :] = density_map_list_9patches[l][5][height//8:(height//2-height//4+height//8), weight//8:(weight-weight//2), :]

                density_map_full_list[l][0][(height//2+height//8):height, 0:(weight//4+weight//8), :] = density_map_list_9patches[l][6][height//8:(height-height//2), 0:(weight//4+weight//8), :]
                density_map_full_list[l][0][(height//2+height//8):height, (weight//4+weight//8):(weight//2+weight//8), :] = density_map_list_9patches[l][7][height//8:(height-height//2), weight//8:(weight//2-weight//4+weight//8), :]
                density_map_full_list[l][0][(height//2+height//8):height, (weight//2+weight//8):weight, :] = density_map_list_9patches[l][8][height//8:(height-height//2), weight//8:(weight-weight//2), :]

            metric_validate, gt_counting, est_counting = sess.run([eval_metric_ops, ground_truth_counting, estimated_counting_2], feed_dict={density_map_list: density_map_full_list, y: label_imgs, phase_tensor: 'test'})

            MAE_validate_.append(metric_validate['MAE'])
            MSE_validate_.append(metric_validate['MSE'])

            axes[0,0].imshow(gt_imgs[0][:, :, ::-1]) #　BGR to RGB
            axes[0,0].set_title('origin Image')
            axes[0,1].imshow(np.squeeze(label_imgs), cmap=plt.cm.jet)
            axes[0,1].set_title('ground_truth {}'.format(gt_counting))
            axes[0,2].imshow(np.squeeze(density_map_full_list[2]), cmap=plt.cm.jet)
            axes[0,2].set_title('estimated_density_map {}'.format(est_counting))
            plt.savefig(density_map_path)
            plt.cla()

        MAE_validate_mean = sess.run(h5, feed_dict={z5: MAE_validate_})
        RMSE_validate = sess.run(h6, feed_dict={z6: MSE_validate_})

        # visualize metrics
        print('In epoch {}, MAE = {}, MSE = {}\r'.format(i, MAE_validate_mean, RMSE_validate))



            
            
