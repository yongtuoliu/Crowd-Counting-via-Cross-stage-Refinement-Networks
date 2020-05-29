import tensorflow as tf
from counting_model import refinement_net



class CountingNet(object):
 
    def __init__(self, phase):
        self._phase = phase
        self._refinement_gan = refinement_net.GenerativeNet(self._phase)


    def build(self, input_tensor, input_tensor_2, input_tensor_4, input_tensor_8, name):
        """
        生成最后的结果图
        """
        with tf.variable_scope(name):
            refinement_lstm_output = self._refinement_gan.build_refinement_lstm(input_tensor=input_tensor, input_tensor_2=input_tensor_2, input_tensor_4=input_tensor_4, input_tensor_8=input_tensor_8, name='refinement_lstm_inference')
            return refinement_lstm_output


