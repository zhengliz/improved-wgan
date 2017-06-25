import os
import sys

import matplotlib
import numpy as np
import theano
import theano.tensor as T

sys.path.append(os.getcwd())
matplotlib.use('Agg')

NUM_SAMPLES = 10000


def get_nll(x, parzen, batch_size=10):
  inds = range(x.shape[0])
  n_batches = int(np.ceil(float(len(inds)) / batch_size))
  nlls = []
  for i in range(n_batches):
    nll = parzen(x[inds[i::n_batches]])
    nlls.extend(nll)

  return np.mean(nlls), np.std(nlls)


def log_mean_exp(a):
  max_ = a.max(1)

  return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
  x = T.matrix()
  mu = theano.shared(mu)
  a = (x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1)) / sigma
  E = log_mean_exp(-0.5 * (a ** 2).sum(2))
  Z = mu.shape[1] * T.log(sigma * np.sqrt(np.pi * 2))

  return theano.function([x], E - Z)


def cross_validate_sigma(samples, data, sigmas, batch_size):
  lls = []
  for sigma in sigmas:
    print "Sigma {}:".format(sigma),
    parzen = theano_parzen(samples, sigma)
    ll_mean, ll_std = get_nll(data, parzen, batch_size=batch_size)
    lls.append(ll_mean)
    print ll_mean, ll_std
    del parzen
  ind = np.argmax(lls)
  return sigmas[ind]


if __name__ == '__main__':
  from gan_mnist_inv import *

  saver = tf.train.Saver(max_to_keep=1000)

  # Train loop
  with tf.Session() as session:
    # load model
    saver = tf.train.import_meta_graph(
      os.path.join(OUTPUT_PATH, 'models/mnist/model-39999.meta'))
    saver.restore(session, os.path.join(OUTPUT_PATH,
                                        'models/mnist/model-39999'))

    # load data, dev and test
    _, valid, test = lib.mnist.load_data()

    # generate samples
    gen_samples = Generator(NUM_SAMPLES).eval()

    # cross validate sigma
    sigma_range = np.logspace(-1., 0., 10)
    sigma = cross_validate_sigma(gen_samples, valid[0], sigma_range, BATCH_SIZE)
    print "Using Sigma: {}".format(sigma)

    # fit and evaulate
    parzen = theano_parzen(gen_samples, sigma)
    ll_mean, ll_std = get_nll(test[0], parzen, BATCH_SIZE)
    ll_std /= np.sqrt(test[0].shape[0])
    print "Log-Likelihood of test set = {}, se: {}".format(ll_mean, ll_std)
