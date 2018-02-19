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

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
#flags.DEFINE_integer('epochs', 50, 'The number of training epochs')
flags.DEFINE_integer('epochs', 200, 'The number of training epochs')
#flags.DEFINE_integer('batchsize', 30, 'The batch size')
flags.DEFINE_integer('batchsize', 10, 'The batch size')
flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')

def load_textual_data(path, train_frac=1., nonzero_frac=0.1):
    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    data = DataFromTxt(path, nonzero_frac)
    if train_frac < 1.:
        data.splitTrainTest(train_frac)
    return Datasets(train=data.getTrainData(), validation=None, test=data.getTestData())

# ensure output dir exists
if not os.path.isdir('out'):
  os.mkdir('out')

### Retrieve data - MNIST
#dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
##trX, trY, teX, teY = dataset.train.images, dataset.train.labels, dataset.test.images, dataset.test.labels
#trX, teX = dataset.train.images, dataset.test.images
#trX, teX = min_max_scale(trX, teX)

### Retrieve data - text
#dataset = load_textual_data("data/docVectors-NASA.out", 0.9, 0.01)
dataset = load_textual_data("data/sentenceVectors-Emails-January.out", 0.9, 0.1)
trX, teX = dataset.train.getNumpyRepresentation(), dataset.test.getNumpyRepresentation()

### RBMs
#visualise = True
#print_step = 100
#architecture = [784, 900, 500, 250, 2] # MNIST
visualise = False
print_step = 100
#architecture = [2000, 500, 250, 100, 20]
architecture = [60, 140, 40, 30, 20]
rbmobject1 = RBM(architecture[0], architecture[1], ['rbmw1', 'rbvb1', 'rbmhb1'], 0.3)
rbmobject2 = RBM(architecture[1], architecture[2], ['rbmw2', 'rbvb2', 'rbmhb2'], 0.3)
rbmobject3 = RBM(architecture[2], architecture[3], ['rbmw3', 'rbvb3', 'rbmhb3'], 0.3)
rbmobject4 = RBM(architecture[3], architecture[4], ['rbmw4', 'rbvb4', 'rbmhb4'], 0.3)

if FLAGS.restore_rbm:
  rbmobject1.restore_weights('./out/rbmw1.chp')
  rbmobject2.restore_weights('./out/rbmw2.chp')
  rbmobject3.restore_weights('./out/rbmw3.chp')
  rbmobject4.restore_weights('./out/rbmw4.chp')

### Autoencoder
autoencoder = AutoEncoder(architecture[0], architecture[1:], [['rbmw1', 'rbmhb1'],
                                                    ['rbmw2', 'rbmhb2'],
                                                    ['rbmw3', 'rbmhb3'],
                                                    ['rbmw4', 'rbmhb4']], tied_weights=False)

iterations = len(trX) / FLAGS.batchsize
print(" Total iterations for batch size %d: %d" % (FLAGS.batchsize, iterations))

# Train First RBM
print('\n > First rbm')
for i in range(FLAGS.epochs):
  print("   Epoch %d" % i)
  for j in range(iterations):
    batch_xs, batch_ys = dataset.train.next_batch(FLAGS.batchsize)
    cost = rbmobject1.partial_fit(batch_xs)
    if j % print_step == 0:
        print("     Iter %d -> cost = %f" % (j, cost))
  print("     -> epoch finished, training set cost: % f" % rbmobject1.compute_cost(trX))
  if visualise:
    show_image("out/1rbm.jpg", rbmobject1.n_w, (28, 28), (30, 30))
rbmobject1.save_weights('./out/rbmw1.chp')

# Train Second RBM2
print('\n > Second rbm')
for i in range(FLAGS.epochs):
  print("   Epoch %d" % i)
  for j in range(iterations):
    batch_xs, batch_ys = dataset.train.next_batch(FLAGS.batchsize)
    # Transform features with first rbm for second rbm
    batch_xs = rbmobject1.transform(batch_xs)
    cost = rbmobject2.partial_fit(batch_xs)
    if j % print_step == 0:
        print("     Iter %d -> cost = %f" % (j, cost))
  print("     -> epoch finished, training set cost: % f" % rbmobject2.compute_cost(rbmobject1.transform(trX)))
  if visualise:
    show_image("out/2rbm.jpg", rbmobject2.n_w, (30, 30), (25, 20))
rbmobject2.save_weights('./out/rbmw2.chp')

# Train Third RBM
print('\n > Third rbm')
for i in range(FLAGS.epochs):
  print("   Epoch %d" % i)
  for j in range(iterations):
    # Transform features
    batch_xs, batch_ys = dataset.train.next_batch(FLAGS.batchsize)
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    cost = rbmobject3.partial_fit(batch_xs)
    if j % print_step == 0:
        print("     Iter %d -> cost = %f" % (j, cost))
  print("     -> epoch finished, training set cost: % f" % rbmobject3.compute_cost(rbmobject2.transform(rbmobject1.transform(trX))))
  if visualise:
    show_image("out/3rbm.jpg", rbmobject3.n_w, (25, 20), (25, 10))
rbmobject3.save_weights('./out/rbmw3.chp')

# Train Fourth RBM
print('\n > Fourth rbm')
for i in range(FLAGS.epochs):
  print("   Epoch %d" % i)
  for j in range(iterations):
    batch_xs, batch_ys = dataset.train.next_batch(FLAGS.batchsize)
    # Transform features
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    batch_xs = rbmobject3.transform(batch_xs)
    cost = rbmobject4.partial_fit(batch_xs)
    if j % print_step == 0:
        print("     Iter %d -> cost = %f" % (j, cost))
  print("     -> epoch finished, training set cost: % f" % rbmobject4.compute_cost(rbmobject3.transform(rbmobject2.transform(rbmobject1.transform(trX)))))
rbmobject4.save_weights('./out/rbmw4.chp')


# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights('./out/rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
autoencoder.load_rbm_weights('./out/rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)
autoencoder.load_rbm_weights('./out/rbmw3.chp', ['rbmw3', 'rbmhb3'], 2)
autoencoder.load_rbm_weights('./out/rbmw4.chp', ['rbmw4', 'rbmhb4'], 3)

# Train Autoencoder
print('\n > Fine-tuning the autoencoder')
for i in range(FLAGS.epochs):
  print("   Epoch %d" % i)
  cost = 0.0
  for j in range(iterations):
    batch_xs, batch_ys = dataset.train.next_batch(FLAGS.batchsize)
    batch_cost = autoencoder.partial_fit(batch_xs)
    cost += batch_cost
    if j % print_step == 0:
        print("     Iter %d -> cost = %f" % (j, batch_cost))
  print("     -> epoch finished")
  print("          sum of costs: %f" % cost)
  print("          avg cost: %f" % (cost / iterations))

autoencoder.save_weights('./out/au.chp')
autoencoder.load_weights('./out/au.chp')

fig, ax = plt.subplots()

if autoencoder.architecture[-1] == 2:
    print("\n > Test set - x coordinates:")
    print(autoencoder.transform(teX)[:, 0])
    print("\n > Test set - y coordinates:")
    print(autoencoder.transform(teX)[:, 1])
    plt.scatter(autoencoder.transform(teX)[:, 0], autoencoder.transform(teX)[:, 1], alpha=0.5)
    plt.show()

# Auto-encoder tests
print("\n > Auto-encoder tests (test set):")
print("     test data shape: %d, %d" % (teX.shape[0], teX.shape[1]))
teX_reco, mse, cosSim = autoencoder.reconstruct(teX)
if visualise:
    plot.plotSubset(autoencoder, teX, teX_reco, n=10, name="testSet", outdir="out/")
print("\n   Input:")
for row in teX[:10]:
    print("    " + ", ".join([repr(el) for el in row[:20]]) + " ...")
print("\n   Prediction:")
for row in teX_reco[:10]:
    print("    " + ", ".join([repr(int(el*10)/10.) for el in row[:20]]) + " ...")
print("\n   MSE: {}".format(mse))
print("   sqrt(MSE): {}".format(mse**0.5))
print("   cosSim: {}".format(cosSim))

raw_input("Press Enter to continue...")
plt.savefig('out/myfig')

