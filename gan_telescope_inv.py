import os
import sys
import time

import matplotlib
import numpy as np
import sklearn.preprocessing
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.plot

sys.path.append(os.getcwd())
matplotlib.use('Agg')

MODE = 'wgan-gp'
BATCH_SIZE = 50
CRITIC_ITERS = 5
LAMBDA = 10
ITERS = 200000
NOISE_DIM = 8
OUTPUT_DIM = 10
DIM = 10
DATA_PATH = '../../Data/telescope'

lib.print_model_settings(locals().copy())

# Load data
print "Loading MAGIC Gamma Telescope Data Set ..."
data = np.genfromtxt(fname=os.path.join(DATA_PATH, 'magic04.data'),
                     dtype='<U20', delimiter=',')
np.random.shuffle(data)
labels = data[:, -1]
data = data[:, :-1].astype(float)
# scale data
scaler = sklearn.preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
# encode binary labels
le = sklearn.preprocessing.LabelEncoder().fit(labels)
labels = le.transform(labels)
# prepare data batches
X_train = data[:16000].reshape(-1, BATCH_SIZE, OUTPUT_DIM)
X_test = data[16000:19000].reshape(-1, BATCH_SIZE, OUTPUT_DIM)
y_train = labels[:16000].reshape(-1, BATCH_SIZE)
y_test = labels[16000:19000].reshape(-1, BATCH_SIZE)
print "Finish loading MAGIC Gamma Telescope Data Set."


def gen(X, y):
  for idx in xrange(len(y)):
    yield np.copy(X[idx]), np.copy(y[idx])


def inf_train_gen():
  while True:
    for instances, targets in gen(X_train, y_train):
      yield instances


def LeakyReLU(x, beta=0.2):
  return tf.maximum(beta * x, x)


def ReLULayer(name, n_in, n_out, inputs):
  output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs,
                                 initialization='he')
  return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
  output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs,
                                 initialization='he')
  return LeakyReLU(output)


def Generator(n_samples, noise):
  if noise is None:
    noise = tf.random_normal([n_samples, NOISE_DIM])

  output = ReLULayer('Generator.Input', NOISE_DIM, DIM, noise)
  output = ReLULayer('Generator.2', DIM, DIM, output)
  output = ReLULayer('Generator.3', DIM, DIM, output)
  output = lib.ops.linear.Linear('Generator.Output', DIM, OUTPUT_DIM, output)

  return output


def Discriminator(inputs):
  output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, DIM, inputs)
  output = LeakyReLULayer('Discriminator.2', DIM, DIM, output)
  output = LeakyReLULayer('Discriminator.3', DIM, DIM, output)

  discriminator_output = lib.ops.linear.Linear('Discriminator.Output', DIM, 1,
                                               output)

  invertor_output = lib.ops.linear.Linear('Invertor.Output', DIM, NOISE_DIM,
                                          output)

  return discriminator_output, invertor_output


# Build graph
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
input_noise = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NOISE_DIM])
fake_data = Generator(BATCH_SIZE, input_noise)

dis_real, _ = Discriminator(real_data)
dis_fake, invert_noise = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
dis_params = lib.params_with_name('Discriminator')
inv_params = lib.params_with_name('Invertor')

# Optimize cost function
if MODE == 'wgan-gp':
  inv_cost = tf.reduce_mean(
    tf.reduce_sum(tf.square(input_noise - invert_noise), axis=1))
  gen_cost = -tf.reduce_mean(dis_fake)
  dis_cost = tf.reduce_mean(dis_fake) - tf.reduce_mean(dis_real)

  alpha = tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
  differences = fake_data - real_data
  interpolates = real_data + alpha * differences
  gradients = tf.gradients(Discriminator(interpolates)[0], [interpolates])[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
  gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
  dis_cost += LAMBDA * gradient_penalty

  inv_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,
                                        beta2=0.9).minimize(inv_cost,
                                                            var_list=inv_params)
  gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,
                                        beta2=0.9).minimize(gen_cost,
                                                            var_list=gen_params)
  dis_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,
                                        beta2=0.9).minimize(dis_cost,
                                                            var_list=dis_params)
  clip_dis_weights = None

# For saving samples
fixed_noise = tf.constant(
  np.random.normal(size=(128, NOISE_DIM)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)

saver = tf.train.Saver(max_to_keep=1000)

# Train loop
with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  for iteration in xrange(ITERS):
    start_time = time.time()
    _input_noise = np.random.normal(size=(BATCH_SIZE, NOISE_DIM))

    _dis_cost = []
    for i in xrange(CRITIC_ITERS):
      _data = inf_train_gen().next()
      _dis_cost_, _ = session.run([dis_cost, dis_train_op],
                                  feed_dict={real_data: _data,
                                             input_noise: _input_noise})
      _dis_cost.append(_dis_cost_)
      if clip_dis_weights:
        _ = session.run(clip_dis_weights)
    _dis_cost = np.mean(_dis_cost)

    _ = session.run(gen_train_op, feed_dict={input_noise: _input_noise})
    _inv_cost, _ = session.run([inv_cost, inv_train_op],
                               feed_dict={input_noise: _input_noise})

    lib.plot.plot('train discriminator cost', _dis_cost)
    lib.plot.plot('train invertor cost', _inv_cost)
    lib.plot.plot('time', time.time() - start_time)

    if iteration % 1000 == 999:
      test_dis_costs = []
      for test_instances, _ in gen(X_test, y_test):
        _test_dis_cost = session.run(dis_cost,
                                     feed_dict={real_data: test_instances,
                                                input_noise: _input_noise})
        test_dis_costs.append(_test_dis_cost)
      lib.plot.plot('test discriminator cost', np.mean(test_dis_costs))
      lib.plot.flush()

    lib.plot.tick()
