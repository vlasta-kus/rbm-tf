import os
import tensorflow as tf
import collections

import matplotlib  
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

from rbm import RBM
from au import AutoEncoder
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
        self.flags.DEFINE_integer('epochs', 100, 'The number of training epochs')
        #self.flags.DEFINE_integer('batchsize', 30, 'The batch size')
        self.flags.DEFINE_integer('batchsize', 10, 'The batch size')
        self.flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')
        
        self.learning_rate = 0.01

        self.visualise = False
        self.print_step = 100
        ############################

        # ensure output dir exists
        if not os.path.isdir('out'):
            os.mkdir('out')

        self.buildGraph()

    def buildGraph(self):
        ### RBMs
        #architecture = [784, 900, 500, 250, 2] # MNIST
        #architecture = [2000, 500, 250, 100, 20]
        architecture = [60, 140, 40, 30, 20]
        self.rbmobject1 = RBM(architecture[0], architecture[1], ['rbmw1', 'rbvb1', 'rbmhb1'], 0.3, transfer_function=tf.nn.sigmoid)
        self.rbmobject2 = RBM(architecture[1], architecture[2], ['rbmw2', 'rbvb2', 'rbmhb2'], 0.3, transfer_function=tf.nn.sigmoid)
        self.rbmobject3 = RBM(architecture[2], architecture[3], ['rbmw3', 'rbvb3', 'rbmhb3'], 0.3, transfer_function=tf.nn.sigmoid)
        self.rbmobject4 = RBM(architecture[3], architecture[4], ['rbmw4', 'rbvb4', 'rbmhb4'], 0.3, transfer_function=tf.nn.sigmoid)
        
        if self.FLAGS.restore_rbm:
          self.rbmobject1.restore_weights('./out/rbmw1.chp')
          self.rbmobject2.restore_weights('./out/rbmw2.chp')
          self.rbmobject3.restore_weights('./out/rbmw3.chp')
          self.rbmobject4.restore_weights('./out/rbmw4.chp')
        
        ### Autoencoder
        self.autoencoder = AutoEncoder(architecture[0], architecture[1:], [['rbmw1', 'rbmhb1'],
                                                            ['rbmw2', 'rbmhb2'],
                                                            ['rbmw3', 'rbmhb3'],
                                                            ['rbmw4', 'rbmhb4']], tied_weights=False,
                                                            optimizer=tf.train.AdamOptimizer(self.learning_rate))
        self.rbmobject1.setSummaryWriter(self.autoencoder.getSummaryWriter())
        self.rbmobject2.setSummaryWriter(self.autoencoder.getSummaryWriter())
        self.rbmobject3.setSummaryWriter(self.autoencoder.getSummaryWriter())
        self.rbmobject4.setSummaryWriter(self.autoencoder.getSummaryWriter())

    
    def load_textual_data(self, path, train_frac=1., nonzero_frac=0.1):
        Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
        data = DataFromTxt(path, nonzero_frac)
        if train_frac < 1.:
            data.splitTrainTest(train_frac)
        return Datasets(train=data.getTrainData(), validation=None, test=data.getTestData())
   
    def getDataFromFile(self, fileName): 
        ### Retrieve data - MNIST
        #dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        ##trX, trY, teX, teY = dataset.train.images, dataset.train.labels, dataset.test.images, dataset.test.labels
        #trX, teX = dataset.train.images, dataset.test.images
        #trX, teX = min_max_scale(trX, teX)
        
        ### Retrieve data - text
        return self.load_textual_data(fileName, 0.9, 0.1)


    def train(self, fileName, save=True):
        # Get data from file
        dataset = self.getDataFromFile(fileName)
        trX, teX = dataset.train.getNumpyRepresentation(), dataset.test.getNumpyRepresentation()
        iterations = len(trX) / self.FLAGS.batchsize
        print(" Total iterations for batch size %d: %d" % (self.FLAGS.batchsize, iterations))
        
        # Train First RBM
        print('\n > First rbm')
        tot_updates = 0
        for i in range(self.FLAGS.epochs):
          print("   Epoch %d" % i)
          for j in range(iterations):
            tot_updates += 1
            batch_xs, batch_ys = dataset.train.next_batch(self.FLAGS.batchsize)
            cost = self.rbmobject1.partial_fit(batch_xs, tot_updates)
            if j % self.print_step == 0:
                print("     Iter %d -> cost = %f" % (j, cost))
          print("     -> epoch finished, training set cost: % f" % self.rbmobject1.compute_cost(trX))
          if self.visualise:
            show_image("out/1rbm.jpg", self.rbmobject1.n_w, (28, 28), (30, 30))
        if save:
            self.rbmobject1.save_weights('./out/rbmw1.chp')
        
        # Train Second RBM2
        print('\n > Second rbm')
        tot_updates = 0
        for i in range(self.FLAGS.epochs):
          print("   Epoch %d" % i)
          for j in range(iterations):
            tot_updates += 1
            batch_xs, batch_ys = dataset.train.next_batch(self.FLAGS.batchsize)
            # Transform features with first rbm for second rbm
            batch_xs = self.rbmobject1.transform(batch_xs)
            cost = self.rbmobject2.partial_fit(batch_xs, tot_updates)
            if j % self.print_step == 0:
                print("     Iter %d -> cost = %f" % (j, cost))
          print("     -> epoch finished, training set cost: % f" % self.rbmobject2.compute_cost(self.rbmobject1.transform(trX)))
          if self.visualise:
            show_image("out/2rbm.jpg", self.rbmobject2.n_w, (30, 30), (25, 20))
        if save:
            self.rbmobject2.save_weights('./out/rbmw2.chp')
        
        # Train Third RBM
        print('\n > Third rbm')
        tot_updates = 0
        for i in range(self.FLAGS.epochs):
          print("   Epoch %d" % i)
          for j in range(iterations):
            tot_updates += 1
            # Transform features
            batch_xs, batch_ys = dataset.train.next_batch(self.FLAGS.batchsize)
            batch_xs = self.rbmobject1.transform(batch_xs)
            batch_xs = self.rbmobject2.transform(batch_xs)
            cost = self.rbmobject3.partial_fit(batch_xs, tot_updates)
            if j % self.print_step == 0:
                print("     Iter %d -> cost = %f" % (j, cost))
          print("     -> epoch finished, training set cost: % f" % self.rbmobject3.compute_cost(self.rbmobject2.transform(self.rbmobject1.transform(trX))))
          if self.visualise:
            show_image("out/3rbm.jpg", self.rbmobject3.n_w, (25, 20), (25, 10))
        if save:
            self.rbmobject3.save_weights('./out/rbmw3.chp')
        
        # Train Fourth RBM
        print('\n > Fourth rbm')
        tot_updates = 0
        for i in range(self.FLAGS.epochs):
          print("   Epoch %d" % i)
          for j in range(iterations):
            tot_updates += 1
            batch_xs, batch_ys = dataset.train.next_batch(self.FLAGS.batchsize)
            # Transform features
            batch_xs = self.rbmobject1.transform(batch_xs)
            batch_xs = self.rbmobject2.transform(batch_xs)
            batch_xs = self.rbmobject3.transform(batch_xs)
            cost = self.rbmobject4.partial_fit(batch_xs, tot_updates)
            if j % self.print_step == 0:
                print("     Iter %d -> cost = %f" % (j, cost))
          print("     -> epoch finished, training set cost: % f" % self.rbmobject4.compute_cost(self.rbmobject3.transform(self.rbmobject2.transform(self.rbmobject1.transform(trX)))))
        if save:
            self.rbmobject4.save_weights('./out/rbmw4.chp')
        
        
        # Load RBM weights to Autoencoder
        self.autoencoder.load_rbm_weights('./out/rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
        self.autoencoder.load_rbm_weights('./out/rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)
        self.autoencoder.load_rbm_weights('./out/rbmw3.chp', ['rbmw3', 'rbmhb3'], 2)
        self.autoencoder.load_rbm_weights('./out/rbmw4.chp', ['rbmw4', 'rbmhb4'], 3)
        
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
            self.autoencoder.save_weights('./out/au.chp')
        #self.autoencoder.load_weights('./out/au.chp')
        
        fig, ax = plt.subplots()
        
        if self.autoencoder.architecture[-1] == 2:
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
    
