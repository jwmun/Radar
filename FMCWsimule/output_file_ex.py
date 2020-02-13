import tensorflow as tf
import rnn_transformer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import glob

radar = rnn_transformer.Radar(train=False, make_data=False)
#radar = radar_rnn.Radar(train=False)
#max_length = radar.max_length
max_length = 416
print(max_length)
#real_data_path = sorted(glob.glob('./test_data/real_signal*.txt'))
#imag_data_path = sorted(glob.glob('./test_data/imag_signal*.txt'))
save_path = './distance_save/model_newal'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    radar.saver.restore(sess, save_path)
    for radar_waveform in range(1, 2):
        for radar_data in range(16, 17):
            real_data_path = sorted(glob.glob('./5thyear/test_input/real_signal_%d_%d_*.txt' %(radar_waveform, radar_data)))
            imag_data_path = sorted(glob.glob('./5thyear/test_input/imag_signal_%d_%d_*.txt' %(radar_waveform, radar_data)))
            signal_index = radar_waveform
            random_index = radar_data
            print(signal_index, random_index)

            def get_max_value(inputs):
                mean_value = []
                for signal in inputs:
                    mean_val = np.sqrt(np.sum([i*i for i in signal]))
                    mean_value.append(mean_val)
                return mean_value


            def make_data(filepath):
                with open(filepath) as f:
                    lines = f.readlines()
                data = []
                for si in lines:
                    data.append([float(i) for i in si.strip().split(' ')])
                return data


            def pad_data(inputs, max_length):
                pad_inputs = []

                for signal_input in inputs:
                    if len(signal_input) > max_length:
                        pad_inputs.append(signal_input[:max_length])
                    else:
                        pad_inputs.append(signal_input + [0]*(max_length-len(signal_input)))


                return pad_inputs


            imag_data = []
            real_data = []
            for i in range(8):
                real_data.append(make_data(real_data_path[i]))

            for i in range(8):
                imag_data.append(make_data(imag_data_path[i]))

            real_max = []
            imag_max = []
            for i in range(8):
                real_max.append(get_max_value(real_data[i]))
                imag_max.append(get_max_value(imag_data[i]))


            def normalize(real_input, imag_input):
                norm_input = []
                norm_label = []
                #max_value = self.get_max_value(inputs)
                for channel in range(8):
                    for ramp in range(75):
                        norm_input.append([float(i)/real_max[channel][ramp] for i in real_input[channel][ramp]])
                        norm_label.append([float(i)/imag_max[channel][ramp] for i in imag_input[channel][ramp]])
                return norm_input, norm_label

            norm_real, norm_imag = normalize(real_data, imag_data)
            matlab_real = np.reshape(np.array(pad_data(norm_real, max_length)), (8, 75, max_length))
            matlab_imag = np.reshape(np.array(pad_data(norm_imag, max_length)), (8, 75, max_length))

            real_matlab = []
            imag_matlab = []
            #save_path = './distance_save/saved_model'
            for i in range(8):
                real_matlab.append(sess.run(radar.logits, feed_dict={radar.signal_input: matlab_real[i], radar.rnn_keep_prob: 1, radar.dense_drop_rate: 0}))
                imag_matlab.append(sess.run(radar.logits, feed_dict={radar.signal_input: matlab_imag[i], radar.rnn_keep_prob: 1, radar.dense_drop_rate: 0}))

            def denormalize(real_input, imag_input, max_length):
                norm_real = []
                norm_imag = []
                #max_value = self.get_max_value(inputs)
                for channel in range(8):
                    for ramp in range(75):
                        norm_real.append([float(i)*real_max[channel][ramp] for i in real_input[channel][ramp]])
                        norm_imag.append([float(i)*imag_max[channel][ramp] for i in imag_input[channel][ramp]])
                norm_real = np.transpose(np.reshape(np.array(norm_real),(8, 75, max_length)), (2, 1, 0))
                norm_imag = np.transpose(np.reshape(np.array(norm_imag),(8, 75, max_length)), (2, 1, 0))

                return norm_real, norm_imag

            real_output, imag_output = denormalize(real_matlab, imag_matlab, max_length)
            scipy.io.savemat('./5thyear/test_output/real_%d_%d.mat' % (signal_index, random_index), mdict={'arr': real_output})
            scipy.io.savemat('./5thyear/test_output/imag_%d_%d.mat' % (signal_index, random_index), mdict={'arr': imag_output})