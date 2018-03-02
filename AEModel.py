import os
import collections
import tensorflow as tf
import numpy as np

import matplotlib  
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

from rbm import RBM
from AutoEncoder import AutoEncoder
from utilsnn import show_image, min_max_scale
from DataFromTxt import DataFromTxt
import input_data
import plot


class AEModel:

    def __init__(self):
        ############################
        self.flags = tf.app.flags
        self.FLAGS = self.flags.FLAGS
        self.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
        #self.flags.DEFINE_integer('epochs', 50, 'The number of training epochs')
        self.flags.DEFINE_integer('epochs', 30, 'The number of training epochs')
        #self.flags.DEFINE_integer('batchsize', 30, 'The batch size')
        self.flags.DEFINE_integer('batchsize', 10, 'The batch size')
        self.flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')
        
        self.learning_rate = 0.001
        self.finetune_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #self.finetune_optimizer = tf.train.AdagradOptimizer(self.learning_rate)

        self.visualise = False
        self.model_path = "./out/au.chp"
        self.print_step = 100
        ############################

        # ensure output dir exists
        if not os.path.isdir('out'):
            os.mkdir('out')

        self.buildGraph()

    def buildGraph(self):
        ### RBMs
        #architecture = [784, 900, 500, 250, 2] # MNIST

        #architecture = [2000, 500, 250, 125, 40]
        #architecture = [2000, 500, 200, 80, 2]

        #architecture = [2000, 500, 250, 50] # NASA docs: cos_sim ~ 80% (training set), ~45% (test set) (0.001, Adam)

        """
        self.architecture = [
                        {   'nodes': 2000,
                            'activation': tf.nn.sigmoid,
                            #'activation': tf.nn.softsign,
                            'alpha': 0.3#0.3
                        },
                        {   'nodes': 500,
                            #'activation': tf.nn.sigmoid,
                            #'activation': tf.nn.softsign,
                            #'activation': tf.nn.relu,
                            'alpha': 0.3
                        },
                        {   'nodes': 250,
                            #'activation': tf.nn.sigmoid,
                            #'activation': tf.nn.softsign,
                            #'activation': tf.nn.relu,
                            'alpha': 0.3
                        },
                        {   'nodes': 50,
                            #'activation': tf.nn.sigmoid,
                            #'activation': tf.nn.relu,
                            'alpha': 0.3
                        } # 2000->500->250->50: ~94% train cos_sim, 45% test cos_sim, sigmoid, Adam, cross-entropy (but also MSE)

                        #{   'nodes': 80,
                            #'activation': tf.nn.sigmoid,
                            #'activation': tf.nn.softsign,
                            #'activation': tf.nn.relu,
                        #    'alpha': 0.3
                        #},
                        #{   'nodes': 20,
                            #'activation': tf.nn.sigmoid,
                            #'activation': tf.nn.softsign,
                            #'activation': tf.nn.relu,
                        #    'alpha': 0.3
                        #} # 2000->500->250->80->20: ~74% train cos_sim, ~35% test cos_sim, Adam
                       ]
        """

        #architecture = [2000, 500, 200, 60, 20]
        #architecture = [2000, 500, 200, 30]
        #architecture = [1000, 300, 100, 50]

        #architecture = [60, 140, 40, 30, 10]
        #architecture = [50, 25, 5]      # cos_sim = 69% (learning_rate=0.01, Adam)
        #architecture = [50, 25, 10, 5] # cos_sim = 68% (learning_rate=0.005, Adam)
        #architecture = [50, 100, 40, 20, 5]

        self.rbmobjects = []
        for idx in range(len(self.architecture)-1):
            self.rbmobjects.append(RBM(self.architecture[idx], self.architecture[idx+1], ['rbmw'+repr(idx), 'rbvb'+repr(idx), 'rbmhb'+repr(idx)]))
        
        if self.FLAGS.restore_rbm:
            for idx, obj in zip(range(len(self.architecture)-1), self.rbmobjects):
                obj.restore_weights("./out/rbmw%d.chp" % idx)
        
        ### Autoencoder
        weights_names = []
        for idx in range(len(self.architecture)-1):
            weights_names.append(['rbmw'+repr(idx), 'rbmhb'+repr(idx)])
        self.autoencoder = AutoEncoder(self.architecture, weights_names, tied_weights=False, optimizer=self.finetune_optimizer)

        # share summary writers for visualising in TensorBoard
        for obj in self.rbmobjects:
            obj.setSummaryWriter(self.autoencoder.getSummaryWriter())

    
    def load_textual_data(self, path, train_frac=1., nonzero_frac=0.1):
        Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
        data = DataFromTxt(path, nonzero_frac)
        if train_frac < 1.:
            data.splitTrainTest(train_frac)
        return Datasets(train=data.getTrainData(), validation=None, test=data.getTestData())
   
    def getDataFromFile(self, fileName, train_fraction, vector_nonzero_fraction): 
        ### Retrieve data - MNIST
        #dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        ##trX, trY, teX, teY = dataset.train.images, dataset.train.labels, dataset.test.images, dataset.test.labels
        #trX, teX = dataset.train.images, dataset.test.images
        #trX, teX = min_max_scale(trX, teX)
        
        ### Retrieve data - text
        return self.load_textual_data(fileName, train_fraction, vector_nonzero_fraction)


    def train(self, fileName, train_fraction, vector_nonzero_fraction, save=True):
        # Get data from file
        dataset = self.getDataFromFile(fileName, train_fraction, vector_nonzero_fraction)
        trX, teX = dataset.train.getNumpyRepresentation(), dataset.test.getNumpyRepresentation()
        iterations = len(trX) / self.FLAGS.batchsize
        print(" Total iterations for batch size %d: %d" % (self.FLAGS.batchsize, iterations))
        
        # Pre-training
        for idx, rbm in zip(range(len(self.rbmobjects)), self.rbmobjects):
            self.pretrainRBM(dataset, iterations, rbm, self.rbmobjects[:idx], idx)
        
        # Load RBM weights to Autoencoder
        for idx, rbm in zip(range(len(self.rbmobjects)), self.rbmobjects):
            self.autoencoder.load_rbm_weights("./out/rbmw%d.chp" % idx, ['rbmw'+repr(idx), 'rbmhb'+repr(idx)], idx)
        
        # Train Autoencoder
        print('\n > Fine-tuning the autoencoder')
        for i in range(self.FLAGS.epochs):
          print("   Epoch %d" % i)
          cost = 0.0
          for j in range(iterations):
            batch_xs, batch_ys = dataset.train.next_batch(self.FLAGS.batchsize)
            batch_cost = self.autoencoder.partial_fit(batch_xs)
            cost += batch_cost
            if j % self.print_step == 0:
                print("     Iter %d -> cost = %f" % (j, batch_cost))
          print("     -> epoch finished")
          print("          sum of costs: %f" % cost)
          print("          avg cost: %f" % (cost / iterations))
        
        if save:
            self.autoencoder.save_weights(self.model_path)
        
        fig, ax = plt.subplots()
        
        if self.architecture[-1]['nodes'] == 2:
            print("\n > Test set - x coordinates:")
            print(self.autoencoder.transform(teX)[:, 0])
            print("\n > Test set - y coordinates:")
            print(self.autoencoder.transform(teX)[:, 1])
            plt.scatter(self.autoencoder.transform(teX)[:, 0], self.autoencoder.transform(teX)[:, 1], alpha=0.5)
            plt.show()
        
        # Auto-encoder tests
        print("\n > Auto-encoder tests (test set):")
        print("     test data shape: %d, %d" % (teX.shape[0], teX.shape[1]))
        teX_reco, mse, cosSim = self.autoencoder.reconstruct(teX)
        if self.visualise:
            plot.plotSubset(self.autoencoder, teX, teX_reco, n=10, name="testSet", outdir="out/")
        print("\n   Input:")
        for row in teX[:10]:
            print("    " + ", ".join([repr(el) for el in row[:20]]) + " ...")
        print("\n   Prediction:")
        for row in teX_reco[:10]:
            print("    " + ", ".join([repr(int(el*10)/10.) for el in row[:20]]) + " ...")
        print("\n   MSE: {}".format(mse))
        print("   sqrt(MSE): {}".format(mse**0.5))
        print("   cosSim: {}".format(cosSim))
        
        #raw_input("Press Enter to continue...")
        #plt.savefig('out/myfig')

    def pretrainRBM(self, dataset, iterations, current_rbm, previous_rbm, idx):
        ### previous_rbm = list of all previous RBMs
        print("\n > Pre-training RBM %d" % idx)
        trX = dataset.train.getNumpyRepresentation()
        tot_updates = 0
        for i in range(self.FLAGS.epochs):
            print("   Epoch %d" % i)
            for j in range(iterations):
                tot_updates += 1
                batch_xs, batch_ys = dataset.train.next_batch(self.FLAGS.batchsize)

                # Transform features (propagate through all previous RBMs)
                for obj in previous_rbm:
                    batch_xs = obj.transform(batch_xs)

                cost = current_rbm.partial_fit(batch_xs, tot_updates)
                if j % self.print_step == 0:
                    print("     Iter %d -> cost = %f" % (j, cost))

            trX_propagated = trX
            for obj in previous_rbm:
                trX_propagated = obj.transform(trX_propagated)
            print("     -> epoch finished, training set cost: % f" % current_rbm.compute_cost(trX_propagated))

        #if save:
        current_rbm.save_weights("./out/rbmw%d.chp" % idx)
 
    def predict(self, data_in):
        # Load model        
        self.autoencoder.load_weights(self.model_path)

        print(" Input data shape: (%d, %d)" % (data_in.shape[0], data_in.shape[1]))
        print(np.sum(data_in, axis=1))

        print(" > Applying model to input data ...")
        prediction = self.autoencoder.transform(data_in)
        print(" Output data shape: (%d, %d)" % (prediction.shape[0], prediction.shape[1]))
        print(np.sum(prediction, axis=1))

        return prediction
