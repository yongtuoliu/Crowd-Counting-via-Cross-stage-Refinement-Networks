import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn

from counting_model import cnn_basenet
from counting_model import vgg16

import cv2

import numpy as np


def inception_arg_scope_sanet(weight_decay=4e-4, std=3, batch_norm_var_collection="moving_vars"):
        instance_norm_params = {# "decay": 0.9997, 
            "epsilon": 1e-6,
            "activation_fn": tf.nn.relu,
            "trainable": True,
            "variables_collections": {"beta":None, "gamma":None, "moving_mean":[batch_norm_var_collection], "moving_variance":[batch_norm_var_collection]},
            "outputs_collections": {}
        }
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay), weights_initializer=tf.glorot_uniform_initializer(), activation_fn=tf.nn.relu) as sc:
            return sc

def inception_arg_scope_convlstm(weight_decay=4e-4, std=3, batch_norm_var_collection="moving_vars"):
        instance_norm_params = {# "decay": 0.9997, 
            "epsilon": 1e-6,
            "activation_fn": tf.nn.relu,
            "trainable": True,
            "variables_collections": {"beta":None, "gamma":None, "moving_mean":[batch_norm_var_collection], "moving_variance":[batch_norm_var_collection]},
            "outputs_collections": {}
        }
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay), weights_initializer=tf.glorot_uniform_initializer()) as sc:
            return sc


class GenerativeNet(cnn_basenet.CNNBaseModel):
    def __init__(self, phase):
        super(GenerativeNet, self).__init__()

        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        return tf.equal(self._phase, self._train_phase)

    def _conv_lstm_2(self, input_tensor, input_cell_state, name, reuse = False):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with slim.arg_scope(inception_arg_scope_convlstm()):
                conv_sigmoid_i = slim.conv2d(input_tensor, 32, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)
                conv_sigmoid_f = slim.conv2d(input_tensor, 32, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)
                conv_relu_g = slim.conv2d(input_tensor, 32, [3, 3], 1, "SAME", activation_fn=tf.nn.relu)
                cell_state = conv_sigmoid_f * input_cell_state + conv_sigmoid_i * conv_relu_g

                conv_sigmoid_o = slim.conv2d(input_tensor, 32, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)

                lstm_feats = conv_sigmoid_o * tf.nn.relu(cell_state)

                ret = {
                    'cell_state': cell_state,
                    'lstm_feats': lstm_feats,
                }

        return ret

    def _conv_lstm_4(self, input_tensor, input_cell_state, name, reuse = False):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with slim.arg_scope(inception_arg_scope_convlstm()):
                conv_sigmoid_i = slim.conv2d(input_tensor, 64, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)
                conv_sigmoid_f = slim.conv2d(input_tensor, 64, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)
                conv_relu_g = slim.conv2d(input_tensor, 64, [3, 3], 1, "SAME", activation_fn=tf.nn.relu)
                cell_state = conv_sigmoid_f * input_cell_state + conv_sigmoid_i * conv_relu_g

                conv_sigmoid_o = slim.conv2d(input_tensor, 64, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)

                lstm_feats = conv_sigmoid_o * tf.nn.relu(cell_state)

                ret = {
                    'cell_state': cell_state,
                    'lstm_feats': lstm_feats,
                }

        return ret

    def _conv_lstm_8(self, input_tensor, input_cell_state, name, reuse = False):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with slim.arg_scope(inception_arg_scope_convlstm()):
                conv_sigmoid_i = slim.conv2d(input_tensor, 128, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)
                conv_sigmoid_f = slim.conv2d(input_tensor, 128, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)
                conv_relu_g = slim.conv2d(input_tensor, 128, [3, 3], 1, "SAME", activation_fn=tf.nn.relu)
                cell_state = conv_sigmoid_f * input_cell_state + conv_sigmoid_i * conv_relu_g

                conv_sigmoid_o = slim.conv2d(input_tensor, 128, [3, 3], 1, "SAME", activation_fn=nn.sigmoid)

                lstm_feats = conv_sigmoid_o * tf.nn.relu(cell_state)

                ret = {
                    'cell_state': cell_state,
                    'lstm_feats': lstm_feats,
                }

        return ret
    
    def scale_aggregation_network(self, features, init_cell_state_2, init_cell_state_4, init_cell_state_8, name, reuse=False):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with slim.arg_scope(inception_arg_scope_sanet()):
                # features = tf.divide(features, 255) 
                features = slim.instance_norm(features, epsilon=1e-6)

                features = slim.conv2d(features, 64, [7, 7], 1, "SAME")

                feature_map_encoder = slim.conv2d(features, 64, [3, 3], 1, "SAME")
                feature_map_encoder = slim.conv2d(feature_map_encoder, 64, [3, 3], 1, "SAME")
                feature_map_encoder = slim.max_pool2d(feature_map_encoder, [2, 2], 2, "SAME", scope="max_pooling_4")

                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")
                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")

                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")
                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")
         
                lstm_ret_2 = self._conv_lstm_2(feature_map_encoder, init_cell_state_2, name='conv_lstm_block_2', reuse=tf.AUTO_REUSE) # skip connection 3(before max_pooling)
                feature_map_encoder = slim.max_pool2d(feature_map_encoder, [2, 2], 2, "SAME", scope="max_pooling_3")

                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")
                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")     
                lstm_ret_4 = self._conv_lstm_4(feature_map_encoder, init_cell_state_4, name='conv_lstm_block_4', reuse=tf.AUTO_REUSE) # skip connection 2(before max_pooling)
                feature_map_encoder = slim.max_pool2d(feature_map_encoder, [2, 2], 2, "SAME", scope="max_pooling_2")

                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")
                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")
                lstm_ret_8 = self._conv_lstm_8(feature_map_encoder, init_cell_state_8, name='conv_lstm_block_8', reuse=tf.AUTO_REUSE) # skip connection 1(before max_pooling)
                feature_map_encoder = slim.max_pool2d(feature_map_encoder, [2, 2], 2, "SAME", scope="max_pooling_1")

                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")
                feature_map_encoder = slim.conv2d(feature_map_encoder, 128, [3, 3], 1, "SAME")


                density_map_estimator = slim.conv2d(feature_map_encoder, 128, [5, 5], 1, "SAME")
                density_map_estimator = slim.conv2d_transpose(density_map_estimator, 128, [2, 2], stride=2, scope="transposed_conv_1")
                density_map_estimator = tf.add(density_map_estimator, lstm_ret_8['lstm_feats']) # skip to after transposed_conv

                density_map_estimator = slim.conv2d(density_map_estimator, 64, [5, 5], 1, "SAME")
                density_map_estimator = slim.conv2d_transpose(density_map_estimator, 64, [2, 2], stride=2, scope="transposed_conv_2")
                density_map_estimator = tf.add(density_map_estimator, lstm_ret_4['lstm_feats']) # skip to after transposed_conv

                density_map_estimator = slim.conv2d(density_map_estimator, 32, [5, 5], 1, "SAME")
                density_map_estimator = slim.conv2d_transpose(density_map_estimator, 32, [2, 2], stride=2, scope="transposed_conv_3")
                density_map_estimator = tf.add(density_map_estimator, lstm_ret_2['lstm_feats']) # skip to after transposed_conv

                density_map_estimator = slim.conv2d(density_map_estimator, 16, [5, 5], 1, "SAME")
                density_map_estimator = slim.conv2d_transpose(density_map_estimator, 16, [2, 2], stride=2, scope="transposed_conv_4")

                density_map_estimator = slim.conv2d(density_map_estimator, 16, [5, 5], 1, "SAME")
            density_map_estimator = slim.conv2d(density_map_estimator, 1, [1, 1], 1, "SAME", normalizer_fn=None, normalizer_params=None)
        return density_map_estimator, lstm_ret_2['cell_state'], lstm_ret_4['cell_state'], lstm_ret_8['cell_state']


    

    def build_refinement_lstm(self, input_tensor, input_tensor_2, input_tensor_4, input_tensor_8, name):

        with tf.variable_scope(name):
            
            temp = input_tensor * 0.0
            init_attention_map = temp[:,:,:,0:1]

            temp_2 = input_tensor_2 * 0.0
            init_cell_state_2 = tf.concat([temp_2, temp_2, temp_2, temp_2, temp_2, temp_2, temp_2, temp_2, temp_2, temp_2, temp_2[:,:,:,0:2]], axis=-1) # channel 32

            temp_4 = input_tensor_4 * 0.0
            temp_4_32 = tf.concat([temp_4, temp_4, temp_4, temp_4, temp_4, temp_4, temp_4, temp_4, temp_4, temp_4, temp_4[:,:,:,0:2]], axis=-1) # temp channel 32
            init_cell_state_4 = tf.concat([temp_4_32, temp_4_32], axis=-1) # channel 64

            temp_8 = input_tensor_8 * 0.0
            temp_8_32 = tf.concat([temp_8, temp_8, temp_8, temp_8, temp_8, temp_8, temp_8, temp_8, temp_8, temp_8, temp_8[:,:,:,0:2]], axis=-1) # temp channel 32
            init_cell_state_8 = tf.concat([temp_8_32, temp_8_32, temp_8_32, temp_8_32], axis=-1) # channel 128

            attention_map_list = []

            for i in range(4):
                attention_input = tf.concat((input_tensor, init_attention_map), axis=-1)

                conv_feats, init_cell_state_2, init_cell_state_4, init_cell_state_8 = self.scale_aggregation_network(attention_input, init_cell_state_2, init_cell_state_4, init_cell_state_8, name='sanet_{:d}'.format(i + 1), reuse=tf.AUTO_REUSE) 

                init_attention_map = conv_feats
                attention_map_list.append(conv_feats)

        return attention_map_list

    def compute_refinement_lstm_loss(self, input_tensor, label_tensor, name):

        with tf.variable_scope(name):
            inference_ret = self.build_attentive_rnn(input_tensor=input_tensor,
                                                     name='attentive_inference')
            loss = tf.constant(0.0, tf.float32)
            n = len(inference_ret['attention_map_list'])
            for index, attention_map in enumerate(inference_ret['attention_map_list']):
                mse_loss = tf.pow(0.8, n - index + 1) * \
                           tf.losses.mean_squared_error(labels=label_tensor,
                                                        predictions=attention_map)
                loss = tf.add(loss, mse_loss)

        return loss, inference_ret['final_attention_map']