import glob
import numpy as np
import random
import pickle

def make_data(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    data = []
    for si in lines:
        data.append([float(i) for i in si.strip().split(' ')])
    return data


def make_input():
    files = sorted(glob.glob('../../data/NormalNoise_Input/signal_ta*'))
    return files


def make_label():
    files = sorted(glob.glob('../../data/NormalNoise_Label/no_interfer_signal_*'))
    return files


def make_inputs_and_labels(input_path, label_path):
    inputs = []
    labels = []
    for i in range(len(input_path)):
        input_data = make_data(input_path[i])
        label_data = make_data(label_path[i])
        inputs.extend(input_data)
        labels.extend(label_data)
    return inputs, labels


class data():
    def __init__(self, use_median_filter=True, train=True):
        self.use_median_filter = use_median_filter
        self.input_path = make_input()
        self.label_path = make_label()
        self.inputs, self.labels = make_inputs_and_labels(self.input_path, self.label_path)
        self.max_length = self.test_max_length(self.inputs)
        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)
        self.inputs, self.labels = self.normalize_array(self.inputs, self.labels)
        
        if use_median_filter:
            print('Use median filter')
        else:
            print('Not use median filter')
        if train:
            x = list(range(len(self.inputs)))
            random.shuffle(x)
            self.inputs = self.inputs[x]
            self.labels = self.labels[x]
            print('This is test', self.inputs.shape)

        print('finished loading data!!')
        
    def test_max_length(self, data):
        maxlength = 0
        for i in range(len(data)):
            if len(data[i]) > maxlength:
                maxlength = len(data[i])
        return maxlength    
    
    def median_filter(self, inputs):
        for idx, signal in enumerate(inputs):
            thres = 100*np.median(np.abs(signal))
            signal[np.abs(signal)>thres] = 0
            inputs[idx] = signal
        return inputs
    
    def normalize_array(self, inputs, labels):
        norm_input = []
        norm_label = []
        if self.use_median_filter:
            inputs = self.median_filter(inputs)
        for idx in range(len(inputs)):
            norm_val = np.sqrt(np.sum(inputs[idx]**2))

            norm_input.append(inputs[idx] / norm_val)
            norm_label.append(labels[idx] / norm_val)

        return np.array(norm_input), np.array(norm_label)


#new_data = data()