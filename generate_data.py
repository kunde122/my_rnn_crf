import numpy as np
import random
import collections
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers

class generator(object):
    def __init__(self,ori_path,max_seq_len=280):
        self.batch_id=0
        self.data,self.labels=self.get_sentences(ori_path)
        self.seqlen=[len(sen) for sen in self.data]
        # max_seq_len=max(self.seqlen)
        self.max_seq_len = max_seq_len
        self.word_dict,self.rev_word_dict=self.build_dictionary(self.data)
        self.data=self.text_to_numbers(self.data,self.word_dict)


    def get_sentences(self,path):
        sens = []
        labels = []
        split_list=[]
        with open(path) as fin:
            for line in fin:
                split_list.append(line.strip().split())
        random.shuffle(split_list)
        for split in split_list:
            sen, label = split[:-1], split[-1]
            sens.append(sen)
            labels.append(label)
        return sens, labels

    # Build dictionary of words
    def build_dictionary(self,sentences, vocabulary_size=50000):
        # Turn sentences (list of strings) into lists of words

        words = [x for sublist in sentences for x in sublist]

        # Initialize list of [word, word_count] for each word, starting with unknown
        count = [['RARE', -1]]

        # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

        # Now create the dictionary
        dictionary = {}
        # For each word, that we want in the dictionary, add it, then make it
        # the value of the prior dictionary length
        for word, word_count in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return (dictionary, reverse_dictionary)

    # Turn text data into lists of integers from dictionary
    def text_to_numbers(self,sentences, word_dict):
        # Initialize the returned data
        data = []
        for sentence in sentences:
            sentence_data = []
            # For each word, either use selected index or rare word index
            for word in sentence:
                if word in word_dict:
                    word_ix = word_dict[word]
                else:
                    word_ix = 0
                sentence_data.append(word_ix)
            #padding
            length=len(sentence_data)
            sentence_data += [0 for i in range(self.max_seq_len - length)]
            data.append(sentence_data)
        return (data)



    def next(self, batch_size):
        """
        生成batch_size的样本。
        如果使用完了所有样本，会重新从头开始。
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen



class BiLSTM_Model():
    def __init__(self, lstm_dim_=100,vocab_size=5000,embeddings_size=200, num_tags_=4, lr_=0.001):
        self.lstm_dim = lstm_dim_
        self.num_tags = num_tags_
        self.lr = lr_
        self.initializer=initializers.xavier_initializer()
        # 词嵌入矩阵
        self.embeddings = tf.get_variable("embeddings",initializer=tf.truncated_normal([vocab_size,embeddings_size],stddev=0.05))
        self.x_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_input')
        self.y_target = tf.placeholder(dtype=tf.int32, shape=[None], name='y_target')
        self.seqlen = tf.placeholder(dtype=tf.int32, shape=[None],name='seq_lengths')

        self.logits = self.project_layer_single(self.build_bigru_layer())
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_target))
    def build_model(self):
        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
        (outputs, outputs_state) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                   self.embeddings, dtype=tf.float32)
        x_in_ = tf.concat(outputs, axis=2)
        #将前向和后向输出拼接起来
        #batch_size, max_time, cell_fw.output_size
        return x_in_

    def build_bigru_layer(self):
        embed_ = tf.nn.embedding_lookup(self.embeddings, self.x_input)
        # self.emd = embed_
        lstm_fw_cell = rnn.GRUCell(self.lstm_dim)
        lstm_bw_cell = rnn.GRUCell(self.lstm_dim)
        (outputs, outputs_state) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                   embed_, dtype=tf.float32,sequence_length=self.seqlen)
        x_in_ = tf.concat(outputs, axis=2)
        #将前向和后向输出拼接起来
        #batch_size, cell_fw.output_size
        # x_in_ = tf.concat(outputs_state, axis=1)
        return x_in_
    def project_layer(self, x_in_):
        with tf.variable_scope("project"):
            with tf.variable_scope("hidden"):
                w_tanh = tf.get_variable("w_tanh", [self.lstm_dim*2, self.lstm_dim], initializer=self.initializer,
                                         regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_tanh = tf.get_variable("b_tanh", [self.lstm_dim], initializer=tf.zeros_initializer())
                x_in_ = tf.reshape(x_in_, [-1, self.lstm_dim*2])
                # output = tf.nn.dropout(tf.tanh(tf.add(tf.matmul(x_in_, w_tanh), b_tanh)), self.dropout)
                output = tf.tanh(tf.add(tf.matmul(x_in_, w_tanh), b_tanh))
            with tf.variable_scope("output"):
                w_out = tf.get_variable("w_out", [self.lstm_dim, self.num_tags], initializer=self.initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_out = tf.get_variable("b_out", [self.num_tags], initializer=tf.zeros_initializer())
                pred_ = tf.add(tf.matmul(output, w_out), b_out)
                logits_ = tf.reshape(pred_, [-1, self.num_steps, self.num_tags], name='logits')
        return logits_
    def project_layer_single(self, x_in_):
        with tf.variable_scope("project"):
            with tf.variable_scope("output"):
                w_out = tf.get_variable("w_out", [self.lstm_dim*2, self.num_tags], initializer=self.initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_out = tf.get_variable("b_out", [self.num_tags], initializer=tf.zeros_initializer())
                x_in_ = tf.reshape(x_in_, [-1, self.lstm_dim * 2])
                pred_ = tf.add(tf.matmul(x_in_, w_out), b_out)
                logits_ = tf.reshape(pred_, [-1, self.num_tags], name='logits')
        return logits_
    def loss_layer(self, project_logits):
        from tensorflow.contrib.crf import crf_log_likelihood
        with tf.variable_scope("crf_loss"):
            log_likelihood, trans = crf_log_likelihood(inputs=project_logits, tag_indices=self.y_target,
                                                       transition_params=self.trans, sequence_lengths=self.max_steps)
        return tf.reduce_mean(-log_likelihood)
