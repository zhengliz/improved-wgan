import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

MODE = 'wgan-gp'
DIM = 64
BATCH_SIZE = 50
CRITIC_ITERS = 5
LAMBDA = 10
ITERS = 200000
OUTPUT_DIM = 28*28
NOISE_DIM = 128

lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
  return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
  output = lib.ops.linear.Linear(
    name + '.Linear',
    n_in,
    n_out,
    inputs,
    initialization='he'
  )
  return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
  output = lib.ops.linear.Linear(
    name + '.Linear',
    n_in,
    n_out,
    inputs,
    initialization='he'
  )
  return LeakyReLU(output)


def Generator(n_samples, noise):
  output = lib.ops.linear.Linear('Generator.Input', NOISE_DIM, 4*DIM * 4 * 4, noise)
  if MODE == 'wgan':
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
  output = tf.nn.relu(output)
  output = tf.reshape(output, [-1, 4*DIM, 4, 4])    # 4x4

  output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
  if MODE == 'wgan':
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
  output = tf.nn.relu(output)                       # 8x8
  output = output[:, :, :7, :7]                     # 7x7

  output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
  if MODE == 'wgan':
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
  output = tf.nn.relu(output)                       # 14x14

  output = lib.ops.deconv2d.Deconv2D('Generator.Output', DIM, 1, 5, output)
  output = tf.nn.sigmoid(output)                    # 28x28

  return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs):
  output = tf.reshape(inputs, [-1, 1, 28, 28])      # 28x28

  output = lib.ops.conv2d.Conv2D('Discriminator.Input', 1, DIM, 5, output, stride=2)
  output = LeakyReLU(output)                        # 14x14

  output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
  if MODE == 'wgan':
    output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
  output = LeakyReLU(output)                        # 7x7

  output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
  if MODE == 'wgan':
    output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
  output = LeakyReLU(output)                        # 4x4

  output = tf.reshape(output, [-1, 4*DIM * 4 * 4])

  discriminator_output = lib.ops.linear.Linear('Discriminator.Output', 4*DIM*4*4, 1, output)
  discriminator_output = tf.reshape(discriminator_output, [-1])

  # invertor_output = lib.ops.linear.Linear('Invertor.4', 4*DIM*4*4, 4*DIM*4, output)
  # invertor_output = LeakyReLU(invertor_output)
  # invertor_output = tf.nn.dropout(invertor_output, keep_prob=0.5)
  #
  # invertor_output = lib.ops.linear.Linear('Invertor.Output', 4*DIM*4, NOISE_DIM, invertor_output)

  invertor_output = lib.ops.linear.Linear('Invertor.Output', 4*DIM*4*4, NOISE_DIM, output)
  invertor_output = tf.reshape(invertor_output, [-1, NOISE_DIM])

  return discriminator_output, invertor_output


input_noise = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NOISE_DIM])
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE, input_noise)

# invert_noise = Invertor(fake_data)
disc_real, _ = Discriminator(real_data)
disc_fake, invert_noise = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')
inv_params = lib.params_with_name('Invertor')


if MODE == 'wgan':
  inv_cost = tf.reduce_mean(tf.reduce_sum(tf.square(input_noise - invert_noise), axis=1))
  gen_cost = -tf.reduce_mean(disc_fake)
  disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

  inv_train_op = tf.train.RMSPropOptimizer(
    learning_rate=5e-5
  ).minimize(inv_cost, var_list=inv_params)
  gen_train_op = tf.train.RMSPropOptimizer(
    learning_rate=5e-5
  ).minimize(gen_cost, var_list=gen_params)
  disc_train_op = tf.train.RMSPropOptimizer(
    learning_rate=5e-5
  ).minimize(disc_cost, var_list=disc_params)

  clip_ops = []
  for var in disc_params:
    clip_bounds = [-.01, .01]
    clip_ops.append(
      tf.assign(
        var,
        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
      )
    )
  clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
  inv_cost = tf.reduce_mean(tf.reduce_sum(tf.square(input_noise - invert_noise), axis=1))
  gen_cost = -tf.reduce_mean(disc_fake)
  disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

  alpha = tf.random_uniform(
    shape=[BATCH_SIZE, 1],
    minval=0.,
    maxval=1.
  )
  differences = fake_data - real_data
  interpolates = real_data + alpha * differences
  gradients = tf.gradients(Discriminator(interpolates)[0], [interpolates])[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
  gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
  disc_cost += LAMBDA * gradient_penalty

  inv_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4,
    beta1=0.5,
    beta2=0.9
  ).minimize(inv_cost, var_list=inv_params)
  gen_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4,
    beta1=0.5,
    beta2=0.9
  ).minimize(gen_cost, var_list=gen_params)
  disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4,
    beta1=0.5,
    beta2=0.9
  ).minimize(disc_cost, var_list=disc_params)

  clip_disc_weights = None

elif MODE == 'dcgan':
  inv_cost = tf.reduce_mean(tf.reduce_sum(tf.square(input_noise - invert_noise), axis=1))
  gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    disc_fake,
    tf.ones_like(disc_fake)
  ))

  disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    disc_fake,
    tf.zeros_like(disc_fake)
  ))
  disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    disc_real,
    tf.ones_like(disc_real)
  ))
  disc_cost /= 2.

  inv_train_op = tf.train.AdamOptimizer(
    learning_rate=2e-4,
    beta1=0.5
  ).minimize(inv_cost, var_list=inv_params)
  gen_train_op = tf.train.AdamOptimizer(
    learning_rate=2e-4,
    beta1=0.5
  ).minimize(gen_cost, var_list=gen_params)
  disc_train_op = tf.train.AdamOptimizer(
    learning_rate=2e-4,
    beta1=0.5
  ).minimize(disc_cost, var_list=disc_params)

  clip_disc_weights = None


# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, NOISE_DIM)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame):
  samples = session.run(fixed_noise_samples)
  lib.save_images.save_images(
    samples.reshape((128, 28, 28)),
    'samples/mnist/samples_{}.png'.format(frame)
  )


# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
  while True:
    for images, targets in train_gen():
      yield images


# For sampling around real images
ROWS = 10
for images, targets in train_gen():
  fixed_real_samples = images[:ROWS]
_, noise_mus = Discriminator(fixed_real_samples)
def sample_image(frame):
  mus = session.run(noise_mus)
  extended_noise = []
  for k in xrange(ROWS):
    extended_noise.append(mus[k])
    extended_noise.extend(np.random.multivariate_normal(mean=mus[k],
                                                        cov=np.identity(NOISE_DIM)/10000,
                                                        size=ROWS-2))
  sampled_noise = tf.cast(tf.constant(np.asarray(extended_noise)), tf.float32)
  sampled_noise_samples = Generator(ROWS*(ROWS-1), noise=sampled_noise)
  generated_noise_samples = session.run(sampled_noise_samples)
  samples = []
  for k in xrange(ROWS):
    samples.append(fixed_real_samples[k])
    samples.extend(generated_noise_samples[k*(ROWS-1) : (k+1)*(ROWS-1)])
  lib.save_images.save_images(
    np.reshape(samples, (ROWS*ROWS, 28, 28)),
    'samples/mnist/perturbations_{}.png'.format(frame)
  )


# Train loop
with tf.Session() as session:

  session.run(tf.global_variables_initializer())

  gen = inf_train_gen()

  for iteration in xrange(ITERS):
    start_time = time.time()

    _input_noise = np.random.normal(size=(BATCH_SIZE, NOISE_DIM))
    if iteration > 0:
      _ = session.run(gen_train_op, feed_dict={input_noise: _input_noise})
      _inv_cost, _ = session.run([inv_cost, inv_train_op], feed_dict={input_noise: _input_noise})
      lib.plot.plot('train inv cost', _inv_cost)

    if MODE == 'dcgan':
      disc_iters = 1
    else:
      disc_iters = CRITIC_ITERS
    for i in xrange(disc_iters):
      _data = gen.next()
      _disc_cost, _ = session.run(
        [disc_cost, disc_train_op],
        feed_dict={real_data: _data, input_noise: _input_noise}
      )
      if clip_disc_weights is not None:
        _ = session.run(clip_disc_weights)

    lib.plot.plot('train disc cost', _disc_cost)
    lib.plot.plot('time', time.time() - start_time)

    # Calculate dev loss and generate samples every 1000 iters
    if iteration % 1000 == 999:
      dev_disc_costs = []
      for images, _ in dev_gen():
        _dev_disc_cost = session.run(
          disc_cost,
          feed_dict={real_data: images, input_noise: _input_noise}
        )
        dev_disc_costs.append(_dev_disc_cost)
      lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

      generate_image(iteration)
      sample_image(iteration)

    # Write logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
      lib.plot.flush()


    lib.plot.tick()
