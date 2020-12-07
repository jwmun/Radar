import tensorflow as tf
import preprocess_final
import random
import os
import pickle
import transformer as encode_model
import numpy as np
import math

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden_size', 200, 'number of hidden cell size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('all_data_size', 150000, 'all data size')
flags.DEFINE_integer('train_size', 148000, 'train size')
flags.DEFINE_integer('valid_size', 1000, 'valid_size')
flags.DEFINE_integer('num_epoch', 1000, 'number of epoch')
flags.DEFINE_boolean('use_clipping', True, 'use clipping')
flags.DEFINE_boolean('use_decay', False, 'exponential decay')
flags.DEFINE_boolean('restore', False, 'use restore')
flags.DEFINE_boolean('use_fftloss', False, 'use fft loss')
flags.DEFINE_float('fft_coefficient', 0.1, 'learning rate')
flags.DEFINE_string('gpu_number', "1", 'determine what gpu to use')
flags.DEFINE_string('save_path', '../../save_data/model_fmcw', 'save path')
flags.DEFINE_string('mod', 'train', 'train, null mod')
flags.DEFINE_integer('num_layer', 2, 'number of transformer layer')
flags.DEFINE_integer('num_head', 1, 'number of transformer head')
flags.DEFINE_integer('num_dff', 128, 'number of transformer dff')
flags.DEFINE_integer('drop_rate', 0, 'Drop out rate')
flags.DEFINE_float('rnn_keep_prob', 1, 'rnn drop out rate')
flags.DEFINE_boolean('make_data', True, 'Determine make data')
flags.DEFINE_boolean('median_filter', False, 'Determine make data')
flags.DEFINE_integer('model', 0, 'model_number')
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_number


class Radar:
    def __init__(self, mod=FLAGS.mod, make_data=FLAGS.make_data):
        print(FLAGS.use_fftloss)
        print(type(FLAGS.use_fftloss))
        self.save_path = FLAGS.save_path

        if make_data:
            # IF interference is high, use median filter
            self.datas = preprocess_final.data(use_median_filter=FLAGS.median_filter)
            self.max_length = self.datas.max_length
            self.data_length = len(self.datas.inputs)
            self.train_inputs = self.datas.inputs[:self.data_length - FLAGS.valid_size]
            self.train_labels = self.datas.labels[:self.data_length - FLAGS.valid_size]
            self.valid_inputs = self.datas.inputs[self.data_length - FLAGS.valid_size:]
            self.valid_labels = self.datas.labels[self.data_length - FLAGS.valid_size:]
            print('all data size is', self.data_length)
            print('train size is', self.train_inputs.shape)
            print('valid size is', self.valid_inputs.shape)

        self.max_length = 416
        self.signal_input = tf.placeholder(tf.float32, [None, self.max_length])
        self.signal_label = tf.placeholder(tf.float32, [None, self.max_length])
        # self.fft_signal_label = tf.placeholder(tf.float32, [None, self.max_length])
        self.rnn_keep_prob = tf.placeholder(tf.float32)
        self.dense_drop_rate = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        if FLAGS.model == 0:
            model = self.SelfAttentionModel()
            self.save_path = '../../save_data/model_transformer'
        if FLAGS.model == 1:
            model = self.SelfAttentionModel2()
            self.save_path = '../../save_data/model_base'
        self.optimizer(model)
        self.saver = tf.train.Saver()
        if mod == 'train':
            self.train()

    def SelfAttentionModel(self):
        new_input = tf.expand_dims(self.signal_input, axis=2)

        with tf.variable_scope('GOU1'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2)
 
        sample_encoder = encode_model.Encoder(num_layers=FLAGS.num_layer, d_model=1,
                                              num_heads=FLAGS.num_head,
                                              dff=FLAGS.num_dff)
        sample_encoder_output = sample_encoder(rnn_output, training=False, mask=None)

        concat_output = tf.concat([rnn_output, sample_encoder_output], axis=2)
        concat_output = self.layernorm1(concat_output)


        with tf.variable_scope('GOU2'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, concat_output, dtype=tf.float32)
        rnn_output2 = tf.concat(output, 2)

        sample_encoder2 = encode_model.Encoder(num_layers=FLAGS.num_layer, d_model=1,
                                              num_heads=FLAGS.num_head,
                                              dff=FLAGS.num_dff)
        sample_encoder_output2 = sample_encoder2(rnn_output2, training=False, mask=None)

        concat_output2 = tf.concat([rnn_output2, sample_encoder_output2], axis=2)
        concat_output2 = self.layernorm1(concat_output2)

        with tf.variable_scope('GOU3'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, concat_output2, dtype=tf.float32)
        rnn_output3 = tf.concat(output, 2)

        
        rnn_output3 = tf.expand_dims(rnn_output3, axis=3)
        average_layer3 = tf.layers.average_pooling2d(rnn_output3, [1, rnn_output3.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal
    
    def SelfAttentionModel2(self):
        new_input = tf.expand_dims(self.signal_input, axis=2)

        with tf.variable_scope('GOU1'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2)
 

        with tf.variable_scope('GOU2'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output, dtype=tf.float32)
        rnn_output2 = tf.concat(output, 2)


        with tf.variable_scope('GOU3'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output2, dtype=tf.float32)
        rnn_output3 = tf.concat(output, 2)

        
        rnn_output3 = tf.expand_dims(rnn_output3, axis=3)
        average_layer3 = tf.layers.average_pooling2d(rnn_output3, [1, rnn_output3.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal
    
    def optimizer(self, model):

        if FLAGS.use_clipping:
            self.logits, self.fft_logits = model
            self.loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.logits, self.signal_label)), reduction_indices=1)))
            self.fft_signal_label = tf.math.l2_normalize(tf.abs(tf.signal.rfft(self.signal_label)), axis=1)
            self.fft_loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.fft_logits, self.fft_signal_label)),
                                      reduction_indices=1)))
            self.sum_loss = self.loss + FLAGS.fft_coefficient * self.fft_loss
            if FLAGS.use_fftloss:
                self.final_loss = self.sum_loss
            else:
                self.final_loss = self.loss

            if FLAGS.use_decay:
                self.global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step,
                                                           2000, 0.96, staircase=True)
                optimize = tf.train.AdamOptimizer(learning_rate)
                gvs = optimize.compute_gradients(self.final_loss)

                def ClipIfNotNone(grad):
                    if grad is None:
                        return grad
                    return tf.clip_by_value(grad, -5, 5)

                capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
                self.optimizer = optimize.apply_gradients(capped_gvs, global_step=self.global_step)
            else:
                optimize = tf.train.AdamOptimizer(FLAGS.learning_rate)
                gvs = optimize.compute_gradients(self.final_loss)

                def ClipIfNotNone(grad):
                    if grad is None:
                        return grad
                    return tf.clip_by_value(grad, -5, 5)

                capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
                self.optimizer = optimize.apply_gradients(capped_gvs)


        else:
            self.logits, self.fft_logits = model
            self.loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.logits, self.signal_label)), reduction_indices=1)))
            self.fft_signal_label = tf.math.l2_normalize(tf.abs(tf.signal.rfft(self.signal_label)), axis=1)
            self.fft_loss = tf.reduce_mean(
                tf.sqrt(
                    tf.reduce_sum(tf.square(tf.subtract(self.fft_logits, self.fft_signal_label)),
                                  reduction_indices=1)))
            self.sum_loss = self.loss + FLAGS.fft_coefficient * self.fft_loss
            if FLAGS.use_fftloss:
                self.final_loss = self.sum_loss
            else:
                self.final_loss = self.loss

            if FLAGS.use_decay:
                self.global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step,
                                                           2000, 0.96, staircase=True)
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.final_loss,
                                                                                global_step=self.global_step)
            else:
                self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.final_loss)

    def train(self):
        total_step = self.train_inputs.shape[0] * FLAGS.num_epoch // FLAGS.batch_size
        print('total step is %d' % total_step)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        min_validation_loss = 10
        best_epoch = 0
        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            if FLAGS.restore == True:
                self.saver.restore(sess, self.save_path)
            for step in range(total_step):
                sample = random.sample(range(self.train_inputs.shape[0]), FLAGS.batch_size)
                train_batch = self.train_inputs[sample]
                train_label = self.train_labels[sample]
                sess.run(self.optimizer, feed_dict={self.signal_input: train_batch, self.signal_label: train_label,
                                                    self.rnn_keep_prob: FLAGS.rnn_keep_prob, self.dense_drop_rate: FLAGS.drop_rate})

                if step % 100 == 0:
                    # print('-----------------------------------------------------------')
                    train_loss = sess.run(self.loss,
                                          feed_dict={self.signal_input: train_batch, self.signal_label: train_label,
                                                     self.rnn_keep_prob: FLAGS.rnn_keep_prob, self.dense_drop_rate: 0})
                    fft_train_loss = sess.run(self.fft_loss,
                                              feed_dict={self.signal_input: train_batch,
                                                         self.signal_label: train_label,
                                                         self.rnn_keep_prob: FLAGS.rnn_keep_prob, self.dense_drop_rate: 0})
                    valid_loss = sess.run(self.loss, feed_dict={self.signal_input: self.valid_inputs,
                                                                self.signal_label: self.valid_labels,
                                                                self.rnn_keep_prob: 1,
                                                                self.dense_drop_rate: 0})
                    fft_valid_loss = sess.run(self.fft_loss, feed_dict={self.signal_input: self.valid_inputs,
                                                                        self.signal_label: self.valid_labels,
                                                                        self.rnn_keep_prob: 1,
                                                                        self.dense_drop_rate: 0})


                    print('current step is %d' % step)
                    num_epoch = step * FLAGS.batch_size // self.train_inputs.shape[0]
                    print('current epoch is %d' % (num_epoch))
                    print('')
                    print('train loss is: %f' % train_loss)
                    print('fft train loss is: %f' % fft_train_loss)
                    print('sum_train loss is: %f' % (train_loss + fft_train_loss))
                    print('')
                    print('valid loss is: %f' % valid_loss)
                    print('fft valid loss is: %f' % fft_valid_loss)
                    print('sum_valid real loss is: %f' % (valid_loss + fft_valid_loss))

                    print('minimum valid loss is: {0:0.4f} in epoch {1}'.format(min_validation_loss, best_epoch))

                    if valid_loss < min_validation_loss:
                        best_epoch = num_epoch
                        min_validation_loss = valid_loss
                        self.saver.save(sess, self.save_path)
                        print(self.save_path)
                     
                        print('best model saved!!')



if __name__ == '__main__':
    Radar()