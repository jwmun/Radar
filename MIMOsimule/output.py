import tensorflow as tf
import rnn_transformer

import numpy as np
import scipy.io
import glob

radar = rnn_transformer.Radar(mod='null', make_data=False)
def make_data(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    data = []
    for si in lines:
        data.append([float(i) for i in si.strip().split(' ')])
    temp_data = np.array(data)
    temp_data = temp_data.tolist()
    return temp_data

def make_inputs(input_path):
    inputs = []
    for i in range(len(input_path)):
        input_data = make_data(input_path[i])
        inputs.extend(input_data)
    return np.array(inputs)

def median_filter(inputs):
    print(inputs.shape)
    for idx, signal in enumerate(inputs):
        thres = 100 * np.median(np.abs(signal))
        signal[np.abs(signal) > thres] = 0
        inputs[idx] = signal
    return inputs


def normalize_array(inputs):
    norm_input = []
    inputs = median_filter(inputs)
    norm_value = []
    for idx in range(len(inputs)):
        norm_val = np.sqrt(np.sum(inputs[idx] ** 2))

        norm_input.append(inputs[idx] / norm_val)
        norm_value.append(norm_val)
    return norm_value, np.array(norm_input)


def denormalize(real_input, imag_input, real_norm_value, imag_norm_value):
    norm_real = np.zeros((256,1024))
    norm_imag = np.zeros((256,1024))
    for i in range(1024):
        norm_real[:,i] = real_input[i]*real_norm_value[i]
        norm_imag[:,i] = imag_input[i]*imag_norm_value[i]
    return norm_real, norm_imag

for data_index in range(1,11):
    real_data_path = './test_input/real_interference_%d.txt' %data_index
    imag_data_path = './test_input/imag_interference_%d.txt' %data_index
    real_data_path = sorted(glob.glob(real_data_path))
    imag_data_path = sorted(glob.glob(imag_data_path))
    real_data = make_inputs(real_data_path)
    imag_data = make_inputs(imag_data_path)

    real_norm_value, norm_real = normalize_array(real_data)
    imag_norm_value, norm_imag = normalize_array(imag_data)


    real_matlab = []
    imag_matlab = []

    save_path = './distance_save/model_mimo2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        radar.saver.restore(sess, save_path)
        real_matlab = sess.run(radar.logits, feed_dict={radar.signal_input: norm_real, radar.rnn_keep_prob: 1, radar.dense_drop_rate: 0})
        imag_matlab = sess.run(radar.logits, feed_dict={radar.signal_input: norm_imag, radar.rnn_keep_prob: 1, radar.dense_drop_rate: 0})



    real_output, imag_output = denormalize(real_matlab, imag_matlab, real_norm_value, imag_norm_value)
    print(real_output.shape)
    print(imag_output.shape)
    scipy.io.savemat('./test_output/real_%d.mat' %data_index,  mdict={'arr': real_output})
    scipy.io.savemat('./test_output/imag_%d.mat' %data_index, mdict={'arr': imag_output})