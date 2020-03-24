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
    temp_data = np.transpose(np.array(data))
    temp_data = temp_data.tolist()
    return temp_data


def make_input():
    files = sorted(glob.glob('./data/Input/simule_input/real_interference*'))
    return files


def make_label():
    files = sorted(glob.glob('./data/Label/simule_label/real_interference_label*'))
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
    def __init__(self, use_median_filter= True, train=True):
        self.use_median_filter = use_median_filter
        self.input_path = make_input()
        self.label_path = make_label()
        self.inputs, self.labels = make_inputs_and_labels(self.input_path, self.label_path)
        self.inputs, self.labels = self.normalize(self.inputs, self.labels)
        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)

        with open('max_value.pickle', 'wb') as f:
            pickle.dump(self.max_value, f)
        with open('max_length.pickle', 'wb') as f:
            pickle.dump(self.max_length, f)

        if train:
            x = list(range(len(self.inputs)))
            random.shuffle(x)
            self.inputs = self.inputs[x]
            self.labels = self.labels[x]
            self.inputs = self.input_to_array(self.inputs)
            self.labels = self.input_to_array(self.labels)

            with open('inputs_array.pickle', 'wb') as f:
                pickle.dump(self.inputs, f)
            with open('labels_array.pickle', 'wb') as f:
                pickle.dump(self.labels, f)

            #print('This is test', self.inputs.shape)
            '''
            with open('inputs.pickle', 'wb') as f:
                pickle.dump(self.inputs, f)
            with open('labels.pickle', 'wb') as f:
                pickle.dump(self.labels, f)
            '''

        print('finished loading data!!')



    def median_filter(self, inputs):
        
        for idx, signal in enumerate(inputs):
            thres = 100*np.median(np.abs(signal))
            signal[np.abs(signal)>thres] = 0
            inputs[idx] = signal
        return inputs
    
    def normalize(self, inputs, labels):
        norm_input = []
        norm_label = []

        if self.use_median_filter:
            inputs = self.median_filter(inputs)
        for idx in range(len(inputs)):
            norm_val = np.sqrt(np.sum(inputs[idx]**2))

            norm_input.append(inputs[idx] / norm_val)
            norm_label.append(labels[idx] / norm_val)

        return np.array(norm_input), np.array(norm_label)

    def input_to_array(self, data):
        final_data = []
        for signal in data:
            temp_inputs = np.zeros((128, 8))
            for index in range(0, 8):
                temp_inputs[:, index] = signal[index:1024:8]
            final_data.append(temp_inputs)
        #print(np.array(final_data).shape)
        return np.array(final_data)

if __name__ == '__main__':
    data()