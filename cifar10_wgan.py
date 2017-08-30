import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
# import tflib.inception_score
import tflib.plot

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and extract to the path
DATA_DIR = '/home/zhengliz/Data/cifar10'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')
OUTPUT_PATH = os.getcwd().replace("Repositories", "Output")

DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)
NOISE_DIM = 128

lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)


class Cifar10WGAN(object):
    def __init__(self):
        self.gen_params = self.dis_params = self.inv_params = None
        self.images_int = None

        self.real_data_int = tf.placeholder(tf.int32,
                                            shape=[BATCH_SIZE, OUTPUT_DIM])
        self.real_data = 2 * ((tf.cast(self.real_data_int,
                                       tf.float32) / 255.) - .5)
        self.input_noise = tf.placeholder(tf.float32,
                                          shape=[BATCH_SIZE, NOISE_DIM])
        self.fake_data = self.generate(BATCH_SIZE, self.input_noise)

        self.dis_real, self.real_noise = self.discriminate(self.real_data)
        self.dis_fake, self.invert_noise = self.discriminate(self.fake_data)

        self.real_data_rec = self.generate(BATCH_SIZE, self.real_noise)

        # Standard WGAN loss
        self.inv_cost = tf.reduce_mean(
            tf.square(self.input_noise - self.invert_noise))
        self.inv_cost += tf.reduce_mean(tf.square(
            self.real_data - self.real_data_rec))
        self.gen_cost = -tf.reduce_mean(self.dis_fake)
        self.dis_cost = tf.reduce_mean(self.dis_fake) \
                         - tf.reduce_mean(self.dis_real)

        # Gradient penalty
        alpha = tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
        differences = self.fake_data - self.real_data
        interpolates = self.real_data + alpha * differences
        gradients = tf.gradients(self.discriminate(interpolates)[0],
                                 [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.dis_cost += LAMBDA * gradient_penalty

        self.inv_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.inv_cost, var_list=self.inv_params)
        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.gen_cost, var_list=self.gen_params)
        self.dis_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.dis_cost, var_list=self.dis_params)

        self.fixed_noise = tf.constant(
            np.random.normal(size=(100, NOISE_DIM)).astype('float32'))


    def generate(self, n_samples, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, NOISE_DIM])

        output = lib.ops.linear.Linear('Generator.Input', NOISE_DIM, 4*4*4*DIM, noise)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*DIM, 4, 4])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.Output', DIM, 3, 5,
                                           output)

        output = tf.tanh(output)

        if self.gen_params is None:
            self.gen_params = lib.params_with_name('Generator')

        return tf.reshape(output, [-1, OUTPUT_DIM])


    def discriminate(self, inputs):
        output = tf.reshape(inputs, [-1, 3, 32, 32])

        output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM, 5, output,
                                       stride=2)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
        # output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
        # output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
        output = LeakyReLU(output)

        output = tf.reshape(output, [-1, 4*4*4*DIM])


        discriminator_output = lib.ops.linear.Linear('Discriminator.Output',
                                                     4*4*4*DIM, 1, output)
        discriminator_output = tf.reshape(discriminator_output, [-1])

        if self.dis_params is None:
            self.dis_params = lib.params_with_name('Discriminator')

        inverter_output = lib.ops.linear.Linear('Inverter.4', 4*4*4*DIM,
                                                4*4*DIM, output)
        inverter_output = LeakyReLU(inverter_output)

        inverter_output = lib.ops.linear.Linear('Inverter.5', 4*4*DIM,
                                                4*NOISE_DIM, inverter_output)
        inverter_output = LeakyReLU(inverter_output)

        inverter_output = lib.ops.linear.Linear('Inverter.Output', 4*NOISE_DIM,
                                                NOISE_DIM, inverter_output)
        inverter_output = tf.reshape(inverter_output, [-1, NOISE_DIM])

        if self.inv_params is None:
            self.inv_params = lib.params_with_name('Inverter')

        return discriminator_output, inverter_output


    def train_gen(self, session, _input_noise):
        _ = session.run(self.gen_train_op,
                        feed_dict={self.input_noise: _input_noise})


    def train_dis(self, session, _data, _input_noise):
        _dis_cost_, _ = session.run([self.dis_cost, self.dis_train_op],
                                    feed_dict={self.real_data_int: _data,
                                               self.input_noise: _input_noise})
        return _dis_cost_


    def train_inv(self, session, _data, _input_noise):
        if self.images_int is None:
            self.images_int = _data.copy()
        _inv_cost, _ = session.run([self.inv_cost, self.inv_train_op],
                                   feed_dict={self.real_data_int: _data,
                                              self.input_noise: _input_noise})
        return _inv_cost


    def generate_from_fixed_noise(self, session, frame):
        samples = session.run(self.generate(100, self.fixed_noise))
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        lib.save_images.save_images(
            samples.reshape((100, 3, 32, 32)),
            os.path.join(OUTPUT_PATH,
                         'samples/cifar10/samples_{}.png'.format(frame)))
        return samples


    def generate_from_fixed_images(self, session, frame):
        images_rec = session.run(self.real_data_rec,
                                 feed_dict={self.real_data_int: self.images_int})
        images_int_rec = ((images_rec + 1.) * (255. / 2)).astype('int32')

        samples = np.zeros((BATCH_SIZE * 2, OUTPUT_DIM), dtype=int)
        for i in range(BATCH_SIZE * 2):
            if i & 1:
                samples[i] = images_int_rec[i/2]
            else:
                samples[i] = self.images_int[i/2]
        lib.save_images.save_images(
            samples.reshape((BATCH_SIZE * 2, 3, 32, 32)),
            os.path.join(OUTPUT_PATH,
                         'samples/cifar10/rec_{}.png'.format(frame)))

        return self.images_int, images_int_rec

    # def get_inception_score(self, session):
    #     all_samples = []
    #     for i in xrange(10):
    #         all_samples.append(session.run(self.generate(100)))
    #     all_samples = np.concatenate(all_samples, axis=0)
    #     all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
    #     all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    #     return lib.inception_score.get_inception_score(list(all_samples))


if __name__ == '__main__':
    cifar10Wgan = Cifar10WGAN()

    saver = tf.train.Saver(max_to_keep=1000)

    # Dataset iterators
    train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)

    def inf_train_gen():
        while True:
            for images, _ in train_gen():
                yield images


    # Train loop
    with tf.Session() as session:
        # saver = tf.train.import_meta_graph(
        #     "/home/zhengliz/Output/improved-wgan/models/cifar10/model-99.meta")
        # saver.restore(
        #     session,
        #     "/home/zhengliz/Output/improved-wgan/models/cifar10/model-99")
        writer = tf.summary.FileWriter('./graphs', session.graph)
        session.run(tf.global_variables_initializer())

        gen = inf_train_gen()

        for iteration in xrange(ITERS):
            start_time = time.time()
            input_noise = np.random.normal(size=(BATCH_SIZE, NOISE_DIM))

            # Train critic and inverter
            dis_iters = CRITIC_ITERS
            dis_cost, inv_cost = [], []
            for i in xrange(dis_iters):
                _data = gen.next()
                dis_cost.append(cifar10Wgan.train_dis(session, _data,
                                                     input_noise))
                inv_cost.append(cifar10Wgan.train_inv(session, _data,
                                                      input_noise))
            # Train generator
            if iteration > 0:
                cifar10Wgan.train_gen(session, input_noise)

            lib.plot.plot('train_dis_cost', np.mean(dis_cost))
            lib.plot.plot('train_inv_cost', np.mean(inv_cost))
            lib.plot.plot('time', time.time() - start_time)

            # Calculate inception score every 1K iters
            # if iteration % 1000 == 999:
            #     inception_score = cifar10Wgan.get_inception_score(session)
            #     lib.plot.plot('inception score', inception_score[0])

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 1000 == 997:
                dev_dis_costs = []
                for images, _ in dev_gen():
                    _dev_dis_cost = session.run(
                        cifar10Wgan.dis_cost,
                        feed_dict={cifar10Wgan.real_data_int: images,
                                   cifar10Wgan.input_noise: input_noise})
                    dev_dis_costs.append(_dev_dis_cost)
                lib.plot.plot('dev_dis_cost', np.mean(dev_dis_costs))
                cifar10Wgan.generate_from_fixed_noise(session, iteration)
                cifar10Wgan.generate_from_fixed_images(session, iteration)

            if iteration % 10000 == 9999:
                save_path = saver.save(session, os.path.join(
                    OUTPUT_PATH, 'models/cifar10/model'), global_step=iteration)
                print "Model saved in file: {}".format(save_path)

            # Save logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()

            lib.plot.tick()

    writer.close()