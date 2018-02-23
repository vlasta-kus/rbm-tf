import tensorflow as tf
import numpy as np


class RBM(object):
    def __init__(self, params1, params2, layer_names):
        self.DEFAULT_ALPHA = 1.0
        self.DEFAULT_ACTIVATION_VISIBLE = tf.nn.sigmoid
        self.DEFAULT_ACTIVATION_HIDDEN  = tf.nn.sigmoid

        assert ("nodes" in params1) and ("nodes" in params2), "ERROR: RBM specifications are missing 'nodes'"
        self.n_input = params1['nodes']
        self.n_hidden = params2['nodes']
        activation_visible = (params1['activation'] if 'activation' in params1 else self.DEFAULT_ACTIVATION_VISIBLE)
        activation_hidden = (params2['activation'] if 'activation' in params2 else self.DEFAULT_ACTIVATION_HIDDEN)
        alpha = (params1['alpha'] if 'alpha' in params1 else self.DEFAULT_ALPHA)

        self.layer_names = layer_names
        self.weights = self._initialize_weights()

        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.n_input], name="x_in")
        self.rbm_w = tf.placeholder(tf.float32,[self.n_input, self.n_hidden], name="rbm_w")
        self.rbm_vb = tf.placeholder(tf.float32,[self.n_input], name="rbm_vb_" + layer_names[0])
        self.rbm_hb = tf.placeholder(tf.float32,[self.n_hidden], name="rbm_hb")

        # variables
        # The weights are initialized to small random values chosen from a zero-mean Gaussian with a
        # standard deviation of about 0.01. It is usually helpful to initialize the bias of visible unit
        # i to log[pi/(1?pi)] where pi is the proportion of training vectors in which unit i is on.
        # Otherwise, initial hidden biases of 0 are usually fine.  It is also possible to start the hidden
        # units with quite large negative biases of about ?4 as a crude way of encouraging sparsity.
        self.n_w = np.zeros([self.n_input, self.n_hidden], np.float32)
        self.n_vb = np.zeros([self.n_input], np.float32)
        self.n_hb = np.zeros([self.n_hidden], np.float32)
        self.o_w = np.random.normal(0.0, 0.01, [self.n_input, self.n_hidden])
        self.o_vb = np.zeros([self.n_input], np.float32)
        self.o_hb = np.zeros([self.n_hidden], np.float32)

        # model/training/performing one Gibbs sample.
        # RBM is generative model, which tries to encode in weights the understanding of data.
        # RBMs typically learn better models if more steps of alternating Gibbs sampling are used.
        # 1. Set visible state to training sample(x) and compute hidden state(h0) of data.
        #    Then we have binary units of hidden state computed. It is very important to make these
        #    hidden states binary, rather than using the probabilities themselves. (see Hinton paper)
        self.h0prob = activation_hidden(tf.matmul(self.x, self.rbm_w) + self.rbm_hb)
        self.h0 = self.sample_prob(self.h0prob)
        # 2. Compute new visible state of reconstruction based on computed hidden state reconstruction.
        #    However, it is common to use the probability, instead of sampling a binary value.
        #    So this can be binary or probability(so i choose to not use sampled probability)
        self.v1 = activation_visible(tf.matmul(self.h0prob, tf.transpose(self.rbm_w)) + self.rbm_vb)
        # 3. Compute new hidden state of reconstruction based on computed visible reconstruction.
        #    When hidden units are being driven by reconstructions, always use probabilities without sampling.
        self.h1 = activation_hidden(tf.matmul(self.v1, self.rbm_w) + self.rbm_hb)

        # compute gradients
        self.w_positive_grad = tf.matmul(tf.transpose(self.x), self.h0)
        self.w_negative_grad = tf.matmul(tf.transpose(self.v1), self.h1)

        # stochastic steepest ascent because we need to maximalize log likelihood of p(visible)
        # dlog(p)/dlog(w) = (visible * hidden)_data - (visible * hidden)_reconstruction
        self.update_w = self.rbm_w + alpha * (self.w_positive_grad - self.w_negative_grad) / tf.to_float(tf.shape(self.x)[0])
        self.update_vb = self.rbm_vb + alpha * tf.reduce_mean(self.x - self.v1, 0)
        self.update_hb = self.rbm_hb + alpha * tf.reduce_mean(self.h0prob  - self.h1, 0)

        # sampling functions
        self.h_sample = activation_hidden(tf.matmul(self.x, self.rbm_w) + self.rbm_hb)
        self.v_sample = activation_visible(tf.matmul(self.h_sample, tf.transpose(self.rbm_w)) + self.rbm_vb)

        # cost
        self.err_sum = tf.reduce_mean(tf.square(self.x - self.v_sample))

        # summaries for TensorBoard
        with tf.name_scope("Pretraining_RBM_" + layer_names[0][-1]):
            self.summary_cost = tf.summary.scalar('cost_RBM_' + layer_names[0][-1], self.err_sum)
            self.summary_mean_w = tf.summary.scalar('mean_w__RBM_' + layer_names[0][-1], tf.reduce_mean(self.rbm_w))
            self.summary_mean_vb = tf.summary.scalar('mean_vb__RBM_' + layer_names[0][-1], tf.reduce_mean(self.rbm_vb))
            self.summary_mean_hb = tf.summary.scalar('mean_hb__RBM_' + layer_names[0][-1], tf.reduce_mean(self.rbm_hb))
            self.summary_w_updateToParam = tf.summary.scalar('mean_w_updateToParameters', tf.reduce_mean(self.update_w / self.rbm_w - 1))
            self.summary_vb_updateToParam = tf.summary.scalar('mean_vb_updateToParameters', tf.reduce_mean(self.update_vb / self.rbm_vb - 1))
            self.summary_hb_updateToParam = tf.summary.scalar('mean_hb_updateToParameters', tf.reduce_mean(self.update_hb / self.rbm_hb - 1))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # for TensorBoard
        #self.merged_summaries = tf.summary.merge_all() # merge all the summaries and write them out
        self.logger = None
        #self.logger = tf.summary.FileWriter("log/", self.sess.graph)

    #def __del__(self):
    #    try:
    #        self.logger.flush()
    #        self.logger.close()
    #    except(AttributeError): # not logging
    #        pass
    def setSummaryWriter(self, logger):
        self.logger = logger

    def compute_cost(self, batch):
        # Use it but don?t trust it. If you really want to know what is going on use multiple histograms.
        # Although it is convenient, the reconstruction error is actually a very poor measure of the progress.
        # As the weights increase the mixing rate falls, so decreases in reconstruction error do not
        # necessarily mean that the model is improving. Small increases do not necessarily mean the model
        # is getting worse.
        return self.sess.run(self.err_sum, feed_dict={self.x: batch, self.rbm_w: self.o_w, self.rbm_vb: self.o_vb, self.rbm_hb: self.o_hb})

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def _initialize_weights(self):
        # These weights are only for storing and loading model for tensorflow Saver.
        all_weights = dict()
        all_weights['w'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden], stddev=0.01, dtype=tf.float32),
                                       name=self.layer_names[0])
        all_weights['vb'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name=self.layer_names[1])
        all_weights['hb'] = tf.Variable(tf.random_uniform([self.n_hidden], dtype=tf.float32), name=self.layer_names[2])
        return all_weights

    def transform(self, batch_x):
        return self.sess.run(self.h_sample, {self.x: batch_x, self.rbm_w: self.o_w, self.rbm_vb: self.o_vb, self.rbm_hb: self.o_hb})

    def restore_weights(self, path):
        saver = tf.train.Saver({self.layer_names[0]: self.weights['w'],
                                self.layer_names[1]: self.weights['vb'],
                                self.layer_names[2]: self.weights['hb']})

        saver.restore(self.sess, path)

        self.o_w = self.weights['w'].eval(self.sess)
        self.o_vb = self.weights['vb'].eval(self.sess)
        self.o_hb = self.weights['hb'].eval(self.sess)

    def save_weights(self, path):
        self.sess.run(self.weights['w'].assign(self.o_w))
        self.sess.run(self.weights['vb'].assign(self.o_vb))
        self.sess.run(self.weights['hb'].assign(self.o_hb))
        saver = tf.train.Saver({self.layer_names[0]: self.weights['w'],
                                self.layer_names[1]: self.weights['vb'],
                                self.layer_names[2]: self.weights['hb']})
        save_path = saver.save(self.sess, path)

    def return_weights(self):
        return self.weights

    def return_hidden_weight_as_np(self):
        return self.n_w

    def partial_fit(self, batch_x, tot_updates):
        # 1. always use small ?mini-batches? of 10 to 100 cases.
        #    For big data with lot of classes use mini-batches of size about 10.
        self.n_w, self.n_vb, self.n_hb = self.sess.run([self.update_w, self.update_vb, self.update_hb],
                                                       feed_dict={self.x: batch_x, self.rbm_w: self.o_w, self.rbm_vb: self.o_vb, self.rbm_hb: self.o_hb})

        self.o_w = self.n_w
        self.o_vb = self.n_vb
        self.o_hb = self.n_hb

        #err_sum = self.sess.run(self.err_sum, feed_dict={self.x: batch_x, self.rbm_w: self.n_w, self.rbm_vb: self.n_vb, self.rbm_hb: self.n_hb})
        err_sum, summ_cost, summ_mean_w, summ_mean_vb, summ_mean_hb, summ_w_updToPar, summ_vb_updToPar, summ_hb_updToPar = self.sess.run([self.err_sum, 
                    self.summary_cost, self.summary_mean_w, self.summary_mean_vb, self.summary_mean_hb, self.summary_w_updateToParam, self.summary_vb_updateToParam, self.summary_hb_updateToParam],
                    feed_dict={self.x: batch_x, self.rbm_w: self.n_w, self.rbm_vb: self.n_vb, self.rbm_hb: self.n_hb})
        if self.logger:
            self.logger.add_summary(summ_cost, tot_updates)
            self.logger.add_summary(summ_mean_vb, tot_updates)
            self.logger.add_summary(summ_mean_hb, tot_updates)
            self.logger.add_summary(summ_mean_w, tot_updates)
            self.logger.add_summary(summ_w_updToPar, tot_updates)
            self.logger.add_summary(summ_vb_updToPar, tot_updates)
            self.logger.add_summary(summ_hb_updToPar, tot_updates)

        return err_sum
