import os, sys
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf

import tflib
import tflib.mnist
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
import tflib.plot
import tflib.save_images



OUTPUT_PATH = os.getcwd().replace("Repositories", "Output")
Z_DIM = 64
BATCH_SIZE = 96
DIS_ITERS = 5
ITERS = 200000


def leaky_relu(x, beta=0.2):
  return tf.maximum(beta * x, x)


# Dataset iterator
train_gen, test_gen = tflib.mnist.load_traintest(BATCH_SIZE, BATCH_SIZE)


def inf_train_gen():
  while True:
    for instances, labels in train_gen():   # targets are labels in range(10)
      yield instances


def tf_kl_div(source, target):
  src_mu, src_var = tf.nn.moments(source, axes=[0])
  tgt_mu, tgt_var = tf.nn.moments(target, axes=[0])
  kl = 0.5 * tf.log(tgt_var / src_var) - 0.5 + (src_var + (src_mu - tgt_mu) ** 2) / (2 * tgt_var)
  return kl


class MnistBWGAN(object):
  def __init__(self, x_dim=784, z_dim=64, latent_dim=64, batch_size=96,
               c_gp_x=10., c_gp_z=10., c_re_x=10., c_kd_z=1.):
    self.x_dim = x_dim
    self.z_dim = z_dim
    self.latent_dim = latent_dim
    self.batch_size = batch_size
    self.c_gp_x = c_gp_x
    self.c_gp_z = c_gp_z
    self.c_re_x = c_re_x
    self.c_kd_z = c_kd_z

    self.gen_params = self.dis_x_params = self.inv_params = self.dis_z_params = None

    self.z_p = tf.placeholder(tf.float32, shape=[None, self.z_dim])
    self.x_p = self.generate(self.z_p)

    self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
    self.z = self.invert(self.x)

    self.dis_x = self.discriminate_x(self.x)
    self.dis_x_p = self.discriminate_x(self.x_p)
    self.dis_z = self.discriminate_z(self.z)
    self.dis_z_p = self.discriminate_z(self.z_p)
    self.rec_x = self.generate(self.z)

    self.gen_cost = -tf.reduce_mean(self.dis_x_p)

    self.inv_cost = -tf.reduce_mean(self.dis_z)   # TODO: reconstruction error, KL divergence

    self.rec_err_x = tf.reduce_mean(tf.square(self.x - self.rec_x))
    self.kl_div_z = tf.reduce_mean(tf_kl_div(self.z, self.z_p))
    self.inv_cost_comb = self.inv_cost + self.c_re_x * self.rec_err_x + self.c_kd_z * self.kl_div_z

    self.dis_x_cost = tf.reduce_mean(self.dis_x_p) - tf.reduce_mean(self.dis_x)   # TODO: gradient penalty

    alpha_x = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
    difference_x = self.x_p - self.x
    interpolate_x = self.x + alpha_x * difference_x
    gradient_x = tf.gradients(self.discriminate_x(interpolate_x), [interpolate_x])[0]
    slope_x = tf.sqrt(tf.reduce_sum(tf.square(gradient_x), axis=1))
    gradient_penalty_x = tf.reduce_mean((slope_x - 1.) ** 2)
    self.dis_x_cost_gp = self.dis_x_cost + self.c_gp_x * gradient_penalty_x

    self.dis_z_cost = tf.reduce_mean(self.dis_z) - tf.reduce_mean(self.dis_z_p)   # TODO: gradient penalty

    alpha_z = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
    difference_z = self.z - self.z_p
    interpolate_z = self.z_p + alpha_z * difference_z
    gradient_z = tf.gradients(self.discriminate_z(interpolate_z), [interpolate_z])[0]
    slope_z = tf.sqrt(tf.reduce_sum(tf.square(gradient_z), axis=1))
    gradient_penalty_z = tf.reduce_mean((slope_z - 1.) ** 2)
    self.dis_z_cost_gp = self.dis_z_cost + self.c_gp_z * gradient_penalty_z

    self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
      self.gen_cost, var_list=self.gen_params)
    self.inv_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
      self.inv_cost_comb, var_list=self.inv_params)
    self.dis_x_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
      self.dis_x_cost_gp, var_list=self.dis_x_params)
    self.dis_z_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
      self.dis_z_cost_gp, var_list=self.dis_z_params)

  def generate(self, z_p):
    assert z_p.shape[1] == self.z_dim

    output = tflib.ops.linear.Linear('Generator.Input', self.z_dim, self.latent_dim * 64, z_p)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, self.latent_dim * 4, 4, 4])  # 4 x 4

    output = tflib.ops.deconv2d.Deconv2D('Generator.2', self.latent_dim * 4, self.latent_dim * 2, 5, output)
    output = tf.nn.relu(output)     # 8 x 8
    output = output[:, :, :7, :7]   # 7 x 7

    output = tflib.ops.deconv2d.Deconv2D('Generator.3', self.latent_dim * 2, self.latent_dim, 5, output)
    output = tf.nn.relu(output)     # 14 x 14

    output = tflib.ops.deconv2d.Deconv2D('Generator.Output', self.latent_dim, 1, 5, output)
    output = tf.nn.sigmoid(output)  # 28 x 28

    if self.gen_params is None:
      self.gen_params = tflib.params_with_name('Generator')

    return tf.reshape(output, [-1, self.x_dim])

  def discriminate_x(self, x):
    output = tf.reshape(x, [-1, 1, 28, 28])   # 28 x 28

    output = tflib.ops.conv2d.Conv2D('Discriminator.X.Input', 1, self.latent_dim, 5, output, stride=2)
    output = leaky_relu(output)   # 14 x 14

    output = tflib.ops.conv2d.Conv2D('Discriminator.X.2', self.latent_dim, self.latent_dim * 2, 5, output, stride=2)
    output = leaky_relu(output)   # 7 x 7

    output = tflib.ops.conv2d.Conv2D('Discriminator.X.3', self.latent_dim * 2, self.latent_dim * 4, 5, output, stride=2)
    output = leaky_relu(output)   # 4 x 4
    output = tf.reshape(output, [-1, self.latent_dim * 64])

    output = tflib.ops.linear.Linear('Discriminator.X.Output', self.latent_dim * 64, 1, output)
    output = tf.reshape(output, [-1])

    if self.dis_x_params is None:
      self.dis_x_params = tflib.params_with_name('Discriminator.X')

    return output

  def invert(self, x):
    output = tf.reshape(x, [-1, 1, 28, 28])   # 28 x 28

    output = tflib.ops.conv2d.Conv2D('Inverter.Input', 1, self.latent_dim, 5, output, stride=2)
    output = leaky_relu(output)   # 14 x 14

    output = tflib.ops.conv2d.Conv2D('Inverter.2', self.latent_dim, self.latent_dim * 2, 5, output, stride=2)
    output = leaky_relu(output)   # 7 x 7

    output = tflib.ops.conv2d.Conv2D('Inverter.3', self.latent_dim * 2, self.latent_dim * 4, 5, output, stride=2)
    output = leaky_relu(output)   # 4 x 4
    output = tf.reshape(output, [-1, self.latent_dim * 64])

    output = tflib.ops.linear.Linear('Inverter.4', self.latent_dim * 64, self.latent_dim * 8, output)
    output = leaky_relu(output)

    output = tflib.ops.linear.Linear('Inverter.Output', self.latent_dim * 8, self.z_dim, output)
    output = tf.reshape(output, [-1, self.z_dim])

    if self.inv_params is None:
      self.inv_params = tflib.params_with_name('Inverter')

    return output

  def discriminate_z(self, z):
    assert z.shape[1] == self.z_dim

    output = tflib.ops.linear.Linear('Discriminator.Z.Input', self.z_dim, self.z_dim * 4, z)
    output = leaky_relu(output)

    output = tflib.ops.linear.Linear('Discriminator.Z.2', self.z_dim * 4, self.z_dim, output)
    output = leaky_relu(output)

    output = tflib.ops.linear.Linear('Discriminator.Z.3', self.z_dim, 16, output)
    output = leaky_relu(output)

    output = tflib.ops.linear.Linear('Discriminator.Z.4', 16, 4, output)
    output = leaky_relu(output)

    output = tflib.ops.linear.Linear('Discriminator.Z.Output', 4, 1, output)
    output = tf.reshape(output, [-1])

    if self.dis_z_params is None:
      self.dis_z_params = tflib.params_with_name('Discriminator.Z')

    return output

  def train_gen(self, sess, x, z_p):
    _gen_cost, _ = sess.run([self.gen_cost, self.gen_train_op], feed_dict={self.x: x, self.z_p: z_p})
    return _gen_cost

  def train_dis_x(self, sess, x, z_p):
    _dis_x_cost_gp, _ = sess.run([self.dis_x_cost_gp, self.dis_x_train_op], feed_dict={self.x: x, self.z_p: z_p})
    return _dis_x_cost_gp

  def train_inv(self, sess, x, z_p):
    _inv_cost_comb, _ = sess.run([self.inv_cost_comb, self.inv_train_op], feed_dict={self.x: x, self.z_p: z_p})
    return _inv_cost_comb

  def train_dis_z(self, sess, x, z_p):
    _dis_z_cost_gp, _ = sess.run([self.dis_z_cost_gp, self.dis_z_train_op], feed_dict={self.x: x, self.z_p: z_p})
    return _dis_z_cost_gp

  def generate_from_noise(self, sess, noise, frame):
    samples = sess.run(self.x_p, feed_dict={self.z_p: noise})
    tflib.save_images.save_images(
      samples.reshape((-1, 28, 28)),
      os.path.join(OUTPUT_PATH, 'samples/mnist_bwgan/samples_{}.png'.format(frame)))
    return samples

  def reconstruct_images(self, sess, images, frame):
    reconstructions = sess.run(self.rec_x, feed_dict={self.x: images})
    comparison = np.zeros((images.shape[0] * 2, images.shape[1]), dtype=np.float32)
    for i in xrange(images.shape[0]):
      comparison[2 * i] = images[i]
      comparison[2 * i + 1] = reconstructions[i]
    tflib.save_images.save_images(
      comparison.reshape((-1, 28, 28)),
      os.path.join(OUTPUT_PATH, 'samples/mnist_bwgan/recs_{}.png'.format(frame)))
    return comparison

  def check_inv_obj(self, sess, x, z_p):
    _inv_cost, _rec_err_x, _kl_div_z, _inv_cost_comb = sess.run([self.inv_cost,
                                                                 self.rec_err_x,
                                                                 self.kl_div_z,
                                                                 self.inv_cost_comb],
                                                                feed_dict={self.x: x, self.z_p: z_p})
    print('Inverter Cost: Dis %.4f, Rec %.4f, KLD %.4f, Tol %.4f\n' %
          (_inv_cost, self.c_re_x * _rec_err_x, self.c_kd_z * _kl_div_z, _inv_cost_comb))
    return _inv_cost, _rec_err_x, _kl_div_z, _inv_cost_comb


if __name__ == '__main__':

  mnistBwgan = MnistBWGAN(z_dim=Z_DIM, batch_size=BATCH_SIZE)

  saver = tf.train.Saver(max_to_keep=1000)

  np.random.seed(326)
  fixed_noise = np.random.randn(64, Z_DIM)

  _, _, test_data = tflib.mnist.load_data()
  fixed_images = test_data[0][:32]
  del test_data

  with tf.Session() as session:
    writer = tf.summary.FileWriter(os.path.join(OUTPUT_PATH, 'models/mnist_bwgan/graphs'), session.graph)

    session.run(tf.global_variables_initializer())

    images = noise = gen_cost = inv_cost = dis_x_cost = dis_z_cost = None

    for iteration in xrange(ITERS):

      for i in xrange(DIS_ITERS):

        noise = np.random.randn(BATCH_SIZE, Z_DIM)
        images = inf_train_gen().next()

        dis_x_cost = mnistBwgan.train_dis_x(session, images, noise)
        dis_z_cost = mnistBwgan.train_dis_z(session, images, noise)
        inv_cost = mnistBwgan.train_inv(session, images, noise)
        mnistBwgan.check_inv_obj(session, images, noise)

      gen_cost = mnistBwgan.train_gen(session, images, noise)

      tflib.plot.plot('gen cost', gen_cost)
      tflib.plot.plot('inv cost', inv_cost)
      tflib.plot.plot('dis x cost', dis_x_cost)
      tflib.plot.plot('dis z cost', dis_z_cost)

      if iteration % 100 == 99:
        mnistBwgan.generate_from_noise(session, fixed_noise, iteration)
        mnistBwgan.reconstruct_images(session, fixed_images, iteration)

      if iteration % 10000 == 9999:
        save_path = saver.save(session, os.path.join(
          OUTPUT_PATH, 'models/mnist_bwgan/model'), global_step=iteration)
        print "Model saved in file: {}".format(save_path)

      # write logs out
      if iteration < 5 or iteration % 100 == 99:
        tflib.plot.flush()

      tflib.plot.tick()

  writer.close()



