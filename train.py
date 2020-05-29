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
result_output = open("./result_A", "w")
best_result_output = open("./best_MAE_MSE.txt", "w")
output_density_map = "./output"
batch_size = 1
epoch = 600
loss_c_weight = 0.001
loss_weight = 0.5

if __name__ == "__main__":
    train_dataset = train_data_provider.DataSet(ops.join(dataset_dir, 'train.txt'))
    validate_dataset = validate_data_provider.DataSet(ops.join(dataset_dir, 'test.txt'))

    # declare tensor
    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    x_2 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_2")
    x_4 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_4")
    x_8 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_8")
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="label")
    phase_tensor = tf.placeholder(tf.string, shape=[], name='phase')

    MAE = sys.maxsize
    MSE = sys.maxsize

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


    train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss=loss, global_step=tf.train.get_global_step())
    
    # for visulization, 3600 and 182 are consistent with the training and validating numbers of batches in each epoch
    z1 = tf.placeholder(tf.float32, shape=[3600, 1], name="loss_train") 
    h1 = tf.reduce_mean(z1, name="loss_train_mean")
    z2 = tf.placeholder(tf.float32, shape=[3600], name="MAE_train") 
    h2 = tf.reduce_mean(z2, axis=0, name="MAE_train_mean")
    z3 = tf.placeholder(tf.float32, shape=[3600], name="MSE_train")
    h3 = tf.sqrt(tf.reduce_mean(z3, axis=0), name="RMSE_train")

    z4 = tf.placeholder(tf.float32, shape=[182, 1], name="loss_validate")
    h4 = tf.reduce_mean(z4, name="loss_validate_mean")
    z5 = tf.placeholder(tf.float32, shape=[182], name="MAE_validate")
    h5 = tf.reduce_mean(z5, axis=0, name="MAE_validate_mean")
    z6 = tf.placeholder(tf.float32, shape=[182], name="MSE_validate")
    h6 = tf.sqrt(tf.reduce_mean(z6, axis=0), name="RMSE_validate")

    #　set tf saver
    saver = tf.train.Saver()

    #　set tf summary
    tboard_save_path = './tboard' 
    loss_train_mean_scalar = tf.summary.scalar(name='train_loss', tensor=h1)
    MAE_train_mean_scalar = tf.summary.scalar(name='train_MAE', tensor=h2)
    RMSE_train_scalar = tf.summary.scalar(name='train_MSE', tensor=h3)
    loss_validate_mean_scalar = tf.summary.scalar(name='validate_loss', tensor=h4)
    MAE_validate_mean_scalar = tf.summary.scalar(name='validate_MAE', tensor=h5)
    RMSE_validate_scalar = tf.summary.scalar(name='validate_MSE', tensor=h6)
    summary_ops = tf.summary.merge_all()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # # model restoration
        # weights_path = './checkpoint_dir/counting_epoch256.ckpt'
        # saver.restore(sess=sess, save_path=weights_path)     

        summary_writer = tf.summary.FileWriter(tboard_save_path, sess.graph)       
        for i in range(epoch):
            # train
            loss_train_ = []
            MAE_train_ = []
            MSE_train_ = []
            image_train_num = 3600
            for j in range(image_train_num):
                gt_imgs, gt_imgs_2, gt_imgs_4, gt_imgs_8, label_imgs = train_dataset.next_batch(batch_size)

                sess.run(train_op, feed_dict={x: gt_imgs, x_2:gt_imgs_2, x_4:gt_imgs_4, x_8:gt_imgs_8, y: label_imgs, phase_tensor: 'train'})
                loss_train, metric_train = sess.run([loss, eval_metric_ops], feed_dict={x: gt_imgs, x_2: gt_imgs_2, x_4:gt_imgs_4, x_8: gt_imgs_8, y: label_imgs, phase_tensor: 'train'})
                loss_train_.append(loss_train)
                MAE_train_.append(metric_train['MAE'])
                MSE_train_.append(metric_train['MSE'])

            # validate
            loss_validate_ = []
            MAE_validate_ = []
            MSE_validate_ = []
            image_validate_num = 182
            for k in range(image_validate_num):
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

                loss_validate, metric_validate = sess.run([loss, eval_metric_ops], feed_dict={density_map_list: density_map_full_list, y: label_imgs, phase_tensor: 'test'})
                loss_validate_.append(loss_validate)
                MAE_validate_.append(metric_validate['MAE'])
                MSE_validate_.append(metric_validate['MSE'])
            loss_validate_mean = sess.run(h4, feed_dict={z4: loss_validate_})
            MAE_validate_mean = sess.run(h5, feed_dict={z5: MAE_validate_})
            RMSE_validate = sess.run(h6, feed_dict={z6: MSE_validate_})

            # visualize metrics
            print('In epoch {}, with loss {}, MAE = {}, MSE = {}\r'.format(i, loss_validate_mean, MAE_validate_mean, RMSE_validate))
            result_output.write("epoch: " + str(i) + "  loss:　" + str(loss_validate_mean) + "  MAE: " + str(MAE_validate_mean) +  "  MSE: " + str(RMSE_validate) + "\r\n")
            result_output.flush()

            summary_result = sess.run(summary_ops, feed_dict={z1: loss_train_, z2: MAE_train_, z3: MSE_train_, z4: loss_validate_, z5: MAE_validate_, z6: MSE_validate_})
            summary_writer.add_summary(summary_result, global_step=i)

            # save model
            if MAE > MAE_validate_mean and MSE > RMSE_validate:
                MAE = MAE_validate_mean
                MSE = RMSE_validate
                best_result_output.write("epoch: " + str(i) + "  loss:　" + str(loss_validate_mean) + "  MAE: " + str(MAE_validate_mean) + "  MSE: " + str(RMSE_validate) + "\r\n")
                best_result_output.flush()


                model_save_dir = './checkpoint_dir'
                if not ops.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                model_name = 'counting_epoch{}.ckpt'.format(i)
                model_save_path = ops.join(model_save_dir, model_name)
                saver.save(sess, model_save_path)
            best_result_output.flush()

result_output.close()
best_result_output.close()