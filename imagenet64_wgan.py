import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.small_imagenet
import tflib.ops.layernorm
import tflib.plot

# Download 64x64 ImageNet at http://image-net.org/small/download.php and
# fill in the path to the extracted files here!
DATA_DIR = '/home/zhengliz/Data/imagenet64'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')
OUTPUT_PATH = os.getcwd().replace("Repositories", "Output")

DIM = 64 # Model dimensionality
NOISE_DIM = 128 # latent variable dimension
CRITIC_ITERS = 5 # How many iterations to train the critic for
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 200000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge
N_GPUS = 2 # Number of GPUs
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

lib.print_model_settings(locals().copy())

# Dataset iterator
train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)


def inf_train_gen():
    while True:
        for (images,) in train_gen():
            yield images


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs,
                                   initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs,
                                   initialization='he')
    return LeakyReLU(output)


def Normalize(name, axes, inputs):
    if ('Discriminator' in name):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name, axes, inputs, fused=True)


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs,
                 he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size,
                                   inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:, :, ::2, ::2],
                       output[:, :, 1::2, ::2],
                       output[:, :, ::2, 1::2],
                       output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs,
                 he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:, :, ::2, ::2],
                       output[:, :, 1::2, ::2],
                       output[:, :, ::2, 1::2],
                       output[:, :, 1::2, 1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size,
                                   output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs,
                 he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size,
                                   output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs,
                  resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim,
                                 output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name + '.BN1', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output,
                    he_init=he_init, biases=False)
    output = Normalize(name + '.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output,
                    he_init=he_init)

    return shortcut + output


def GoodGenerator(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, NOISE_DIM])

    output = lib.ops.linear.Linear('Generator.Input', NOISE_DIM, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')

    output = Normalize('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def GoodDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down')

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])



class Imagenet64WGAN():
    def __init__(self):
        self.all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
        split_real_data_conv = tf.split(self.all_real_data_conv, len(DEVICES))

        gen_costs, dis_costs = [], []

        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            with tf.device(device):
                real_data = tf.reshape(
                    2 * ((tf.cast(real_data_conv, tf.float32) / 255.) - .5),
                    [BATCH_SIZE / len(DEVICES), OUTPUT_DIM])
                fake_data = self.generate(BATCH_SIZE / len(DEVICES))

                dis_real = self.discriminate(real_data)
                dis_fake = self.discriminate(fake_data)

                gen_cost = -tf.reduce_mean(dis_fake)
                dis_cost = tf.reduce_mean(dis_fake) - tf.reduce_mean(dis_real)

                alpha = tf.random_uniform(shape=[BATCH_SIZE / len(DEVICES), 1],
                                          minval=0., maxval=1.)
                differences = fake_data - real_data
                interpolates = real_data + alpha * differences
                gradients = tf.gradients(self.discriminate(interpolates),
                                         [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                dis_cost += LAMBDA * gradient_penalty

                gen_costs.append(gen_cost)
                dis_costs.append(dis_cost)

        self.gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        self.dis_cost = tf.add_n(dis_costs) / len(DEVICES)

        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0., beta2=0.9).minimize(
            self.gen_cost, var_list=lib.params_with_name('Generator'),
            colocate_gradients_with_ops=True)
        self.dis_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0., beta2=0.9).minimize(
            self.dis_cost, var_list=lib.params_with_name('Discriminator.'),
            colocate_gradients_with_ops=True)


    def generate(self, n_samples, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, NOISE_DIM])

        output = lib.ops.linear.Linear('Generator.Input', NOISE_DIM,
                                       4 * 4 * 8 * DIM, noise)
        output = tf.reshape(output, [-1, 8 * DIM, 4, 4])

        output = ResidualBlock('Generator.Res1', 8 * DIM, 8 * DIM, 3, output,
                               resample='up')
        output = ResidualBlock('Generator.Res2', 8 * DIM, 4 * DIM, 3, output,
                               resample='up')
        output = ResidualBlock('Generator.Res3', 4 * DIM, 2 * DIM, 3, output,
                               resample='up')
        output = ResidualBlock('Generator.Res4', 2 * DIM, DIM, 3, output,
                               resample='up')

        output = Normalize('Generator.OutputN', [0, 2, 3], output)
        output = tf.nn.relu(output)
        output = lib.ops.conv2d.Conv2D('Generator.Output', DIM, 3, 3, output)
        output = tf.tanh(output)

        return tf.reshape(output, [-1, OUTPUT_DIM])


    def discriminate(self, inputs):
        output = tf.reshape(inputs, [-1, 3, 64, 64])
        output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM, 3, output,
                                       he_init=False)

        output = ResidualBlock('Discriminator.Res1', DIM, 2 * DIM, 3, output,
                               resample='down')
        output = ResidualBlock('Discriminator.Res2', 2 * DIM, 4 * DIM, 3,
                               output, resample='down')
        output = ResidualBlock('Discriminator.Res3', 4 * DIM, 8 * DIM, 3,
                               output, resample='down')
        output = ResidualBlock('Discriminator.Res4', 8 * DIM, 8 * DIM, 3,
                               output, resample='down')

        output = tf.reshape(output, [-1, 4 * 4 * 8 * DIM])
        output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * DIM,
                                       1, output)

        return tf.reshape(output, [-1])


    def train_gen(self, session):
        _ = session.run(self.gen_train_op)
        # to do: add input noise


    def train_dis(self, session, _data):
        _dis_cost, _ = session.run([self.dis_cost, self.dis_train_op],
                                   feed_dict={self.all_real_data_conv: _data})
        return _dis_cost


    def train_inv(self):
        pass


    def generate_from_fixed_noise(self, session, frame):
        # For generating samples
        fixed_noise = tf.constant(
            np.random.normal(size=(BATCH_SIZE, NOISE_DIM)).astype('float32'))
        all_fixed_noise_samples = []
        for device_index, device in enumerate(DEVICES):
            n_samples = BATCH_SIZE / len(DEVICES)
            all_fixed_noise_samples.append(
                self.generate(n_samples,
                              noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))

        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)

        samples = session.run(all_fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)),
                                    os.path.join(OUTPUT_PATH, 'imagenet64/samples_{}.png'.format(frame)))

        return samples


    def generate_from_fixed_images(self, session, frame):
        pass


if __name__ == '__main__':
    imagenet64Wgan = Imagenet64WGAN()

    saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

        # Save a batch of ground-truth samples
        # _x = inf_train_gen().next()
        # _x_r = session.run(
        #     imagenet64Wgan.real_data,
        #     feed_dict={imagenet64Wgan.real_data_conv: _x[:BATCH_SIZE / N_GPUS]})
        # _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')
        # lib.save_images.save_images(
        #     _x_r.reshape((BATCH_SIZE / N_GPUS, 3, 64, 64)),
        #     os.path.join(OUTPUT_PATH, 'imagenet64/samples_groundtruth.png'))


        # Train loop
        session.run(tf.global_variables_initializer())
        gen = inf_train_gen()
        for iteration in xrange(ITERS):

            start_time = time.time()

            # Train generator
            if iteration > 0:
                imagenet64Wgan.train_gen(session)

            dis_iters = CRITIC_ITERS
            for i in xrange(dis_iters):
                _data = gen.next()
                _dis_cost = imagenet64Wgan.train_dis(session, _data)

            lib.plot.plot('train disc cost', _dis_cost)
            lib.plot.plot('time', time.time() - start_time)

            if iteration % 200 == 198:
                t = time.time()
                dev_dis_costs = []
                for (images,) in dev_gen():
                    _dev_dis_cost = session.run(
                        imagenet64Wgan.dis_cost,
                        feed_dict={imagenet64Wgan.all_real_data_conv: images})
                    dev_dis_costs.append(_dev_dis_cost)
                lib.plot.plot('dev disc cost', np.mean(dev_dis_costs))

                imagenet64Wgan.generate_from_fixed_noise(session, iteration)

            if (iteration < 5) or (iteration % 200 == 199):
                lib.plot.flush()

            lib.plot.tick()
