"""
 Model and other utility functions
 Source: https://github.com/nyu-mll/multiNLI
"""

import tensorflow as tf
import numpy as np

def seq_length_tf(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes,
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask

def biLSTM(inputs, dim, seq_len, name,initial_state_fw=None,initial_state_bw=None):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd,
                                                                     inputs=inputs, sequence_length=seq_len,
                                                                     dtype=tf.float32, scope=name,
                                                                     initial_state_fw=initial_state_fw,
                                                                     initial_state_bw=initial_state_bw)

    return hidden_states, cell_states

def LSTM(inputs, dim, seq_len, name):
    """
    An LSTM layer. Returns hidden states and cell states as a tuple.

    Ouput shape of hidden states: (batch_size, max_seq_length, hidden_dim)
    Same shape for cell states.
    """
    with tf.name_scope(name):
        cell = tf.contrib.rnn.LSTMCell(num_units=dim)
        hidden_states, cell_states = tf.nn.dynamic_rnn(cell, inputs=inputs, sequence_length=seq_len,
                                                       dtype=tf.float32, scope=name)

    return hidden_states, cell_states


def last_output(output, true_length):
    """
    To get the last hidden layer form a dynamically unrolled RNN.
    Input of shape (batch_size, max_seq_length, hidden_dim).

    true_length: Tensor of shape (batch_size). Such a tensor is given by the length() function.
    Output of shape (batch_size, hidden_dim).
    """
    max_length = int(output.get_shape()[1])
    length_mask = tf.expand_dims(tf.one_hot(true_length - 1, max_length, on_value=1., off_value=0.), -1)
    last_output = tf.reduce_sum(tf.multiply(output, length_mask), 1)
    return last_output

def masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.

    Input shape: (batch_size, max_seq_length, hidden_dim).
    mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the length() function.
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 1, keep_dims=True))) * mask
    denominator = tf.reduce_sum(numerator, 1, keep_dims=True)
    weights = tf.div(numerator, denominator)
    return weights



def get_minibatch(dataset, start_index, end_index):
    datas = dataset[start_index:end_index]
    premise_vectors = np.vstack([data['premise_index_sequence'] for data in datas])
    hypothesis_vectors = np.vstack([data['hypothesis_index_sequence'] for data in datas])
    return premise_vectors, hypothesis_vectors


class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train, emb_train_topn=None):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        ## Define parameters
        if emb_train_topn:
            E_train = tf.Variable(embeddings[:emb_train_topn], trainable=True)
            E_nontrain = tf.Variable(embeddings[emb_train_topn:], trainable=False)
            self.E = tf.concat([E_train, E_nontrain], 0)
        else:
            self.E = tf.Variable(embeddings, trainable=emb_train)

        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.dim, 2], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([2], stddev=0.1))

        ## Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, mask_prem = seq_length_tf(self.premise_x)
        hyp_seq_lengths, mask_hyp = seq_length_tf(self.hypothesis_x)

        ### First biLSTM layer ###

        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)

        premise_outs, c1 = biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        hypothesis_outs, c2 = biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')



        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        premise_list = tf.unstack(premise_bi, axis=1)
        hypothesis_list = tf.unstack(hypothesis_bi, axis=1)

        ### Attention ###

        scores_all = []
        premise_attn = []
        alphas = []

        for i in range(self.sequence_length):

            scores_i_list = []
            for j in range(self.sequence_length):
                score_ij = tf.reduce_sum(tf.multiply(premise_list[i], hypothesis_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)

            scores_i = tf.stack(scores_i_list, axis=1)
            alpha_i = masked_softmax(scores_i, mask_hyp)
            a_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, hypothesis_bi), 1)
            premise_attn.append(a_tilde_i)

            scores_all.append(scores_i)
            alphas.append(alpha_i)

        scores_stack = tf.stack(scores_all, axis=2)
        scores_list = tf.unstack(scores_stack, axis=1)

        hypothesis_attn = []
        betas = []
        for j in range(self.sequence_length):
            scores_j = scores_list[j]
            beta_j = masked_softmax(scores_j, mask_prem)
            b_tilde_j = tf.reduce_sum(tf.multiply(beta_j, premise_bi), 1)
            hypothesis_attn.append(b_tilde_j)

            betas.append(beta_j)

        # Make attention-weighted sentence representations into one tensor,
        premise_attns = tf.stack(premise_attn, axis=1)
        hypothesis_attns = tf.stack(hypothesis_attn, axis=1)

        # For making attention plots,
        self.alpha_s = tf.stack(alphas, axis=2)
        self.beta_s = tf.stack(betas, axis=2)

        ### Subcomponent Inference ###

        prem_diff = tf.subtract(premise_bi, premise_attns)
        prem_mul = tf.multiply(premise_bi, premise_attns)
        hyp_diff = tf.subtract(hypothesis_bi, hypothesis_attns)
        hyp_mul = tf.multiply(hypothesis_bi, hypothesis_attns)

        m_a = tf.concat([premise_bi, premise_attns, prem_diff, prem_mul], 2)
        m_b = tf.concat([hypothesis_bi, hypothesis_attns, hyp_diff, hyp_mul], 2)

        ### Inference Composition ###
        v1_outs, c3 = biLSTM(m_a, dim=self.dim, seq_len=prem_seq_lengths, name='v1')
        v2_outs, c4 = biLSTM(m_b, dim=self.dim, seq_len=hyp_seq_lengths, name='v2')


        v1_bi = tf.concat(v1_outs, axis=2)
        v2_bi = tf.concat(v2_outs, axis=2)

        ### Pooling Layer ###

        v_1_sum = tf.reduce_sum(v1_bi, 1)
        v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

        v_2_sum = tf.reduce_sum(v2_bi, 1)
        v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)

        # MLP layer
        h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp)

        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        # Define the cost function
        self.total_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))


