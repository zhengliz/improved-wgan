import os, sys
sys.path.append(os.getcwd())

import random
import time
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn
import sklearn.datasets
from sklearn.utils import check_random_state

import tflib as lib
import tflib.ops.linear
import tflib.plot

MODE = 'wgan-gp'  # wgan or wgan-gp
DATASET = 'swissroll'  # 8gaussians, 25gaussians, swissroll
DIM = 512  # Model dimensionality
LAMBDA = .1  # Smaller lambda makes things faster for toy tasks, but isn't
# necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 512  # Batch size
ITERS = 100000  # how many generator iterations to train for
ZDIM = 1

lib.print_model_settings(locals().copy())


STD0 = 0.1
STD1 = 0.1
SCALE = 1.5
def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):

    generator = check_random_state(random_state)

    t = 1.0 * np.pi * (0. + 1.5 * generator.rand(1, n_samples))
    x = t * np.sin(t)
    y = 21 * generator.rand(1, n_samples)
    z = t * np.cos(t)

    X = np.concatenate((x, y, z))
    X += noise * generator.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)

    return X, t



def inf_train_gen():
    while True:
        x0 = make_swiss_roll(
            n_samples=BATCH_SIZE / 2,
            noise=STD0
        )[0][:, [0, 2]]
        x0[:,1] = -x0[:,1]
        x1 = make_swiss_roll(
            n_samples=BATCH_SIZE/2,
            noise=STD1
        )[0][:, [0, 2]]
        x1[:, 0] = -x1[:, 0]
        x = np.vstack((x0, x1)).astype('float32')
        # x /= 20. * np.sqrt((STD0 ** 2 + (STD1 * SCALE) ** 2) / 2.)
        # x /= 2.
        y = np.hstack((np.zeros(shape=(BATCH_SIZE/2,), dtype=int),
                       np.ones(shape=(BATCH_SIZE/2,), dtype=int)))
        indices = range(BATCH_SIZE)
        np.random.shuffle(indices)
        yield x[indices], y[indices]



def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output


def Generator(noise):
    output = ReLULayer('Generator.1', ZDIM, DIM, noise)
    output = ReLULayer('Generator.2', DIM, 2 * DIM, output)
    output = ReLULayer('Generator.3', 2 * DIM, 2 * DIM, output)
    output = ReLULayer('Generator.4', 2 * DIM, DIM, output)
    output = lib.ops.linear.Linear('Generator.5', DIM, 2, output)
    return output


def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 2, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, 2 * DIM, output)
    output = ReLULayer('Discriminator.3', 2 * DIM, 2 * DIM, output)
    output = ReLULayer('Discriminator.4', 2 * DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.5', DIM, 1, output)
    return tf.reshape(output, [-1])


def Inverter(inputs):
    output = ReLULayer('Inverter.1', 2, DIM, inputs)
    output = ReLULayer('Inverter.2', DIM, 2 * DIM, output)
    output = ReLULayer('Inverter.3', 2 * DIM, 2 * DIM, output)
    output = ReLULayer('Inverter.4', 2 * DIM, DIM, output)
    output = lib.ops.linear.Linear('Inverter.5', DIM, ZDIM, output)
    return tf.reshape(output, [-1, ZDIM])


real_data = tf.placeholder(tf.float32, shape=[None, 2])
input_noise = tf.placeholder(tf.float32, shape=[None, ZDIM])

fake_data = Generator(input_noise)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

inv_real = Inverter(real_data)
inv_fake = Inverter(fake_data)

rec_real = Generator(inv_real)

inv_cost = 10. * tf.reduce_mean(tf.square(input_noise - inv_fake))\
           + tf.reduce_mean(tf.abs(input_noise - inv_fake))
           # + 10. * tf.reduce_mean(tf.square(real_data - rec_real))

# WGAN loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN gradient penalty

alpha = tf.random_uniform(
    shape=[BATCH_SIZE, 1],
    minval=0.,
    maxval=1.
)
interpolates = alpha * real_data + ((1 - alpha) * fake_data)
disc_interpolates = Discriminator(interpolates)
gradients = tf.gradients(disc_interpolates, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

disc_cost += LAMBDA * gradient_penalty

disc_params = lib.params_with_name('Discriminator')
gen_params = lib.params_with_name('Generator')
inv_params = lib.params_with_name('Inverter')


global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-4
inv_lr = tf.train.exponential_decay(starter_learning_rate, global_step,
                                    100, 0.99, staircase=True)
inv_train_op = tf.train.AdamOptimizer(
    learning_rate=inv_lr,
    beta1=0.9,
    beta2=0.999
).minimize(
    inv_cost,
    var_list=inv_params
)

disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4,
    beta1=0.5,
    beta2=0.9
).minimize(
    disc_cost,
    var_list=disc_params
)

if len(gen_params) > 0:
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(
        gen_cost,
        var_list=gen_params
    )
else:
    gen_train_op = tf.no_op()

print "Generator params:"
for var in lib.params_with_name('Generator'):
    print "\t{}\t{}".format(var.name, var.get_shape())
print "Discriminator params:"
for var in lib.params_with_name('Discriminator'):
    print "\t{}\t{}".format(var.name, var.get_shape())

frame_index = [0]


def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    # N_POINTS = 128
    # RANGE = 3
    #
    # points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    # points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    # points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    # points = points.reshape((-1,2))
    # samples, disc_map = session.run(
    #     [fake_data, disc_real],
    #     feed_dict={real_data:points}
    # )
    # disc_map = session.run(disc_real, feed_dict={real_data:points})
    #
    # plt.clf()
    #
    # x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    # plt.contour(x,y,disc_map.reshape((len(x), len(y))).transpose())
    #
    # plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    # plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')
    #
    # plt.savefig('frame'+str(frame_index[0])+'.png')
    # frame_index[0] += 1


    _noise_, _rec_ = session.run([inv_real, rec_real],
                                 feed_dict={real_data: true_dist})

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    ax[0].set_title('real data')
    ax[0].set_xlim([-2., 2.])
    ax[0].set_ylim([-2., 2.])

    ax[1].scatter(_noise_[:, 0], _noise_[:, 1], c='green', marker='x')
    ax[1].set_title('latent z')
    ax[1].set_xlim([-2., 2.])
    ax[1].set_ylim([-2., 2.])

    ax[2].scatter(_rec_[:, 0], _rec_[:, 1], c='red', marker='+')
    ax[2].set_title('rec data')
    ax[2].set_xlim([-2., 2.])
    ax[2].set_ylim([-2., 2.])

    fig.savefig('/home/zhengliz/Output/improved-wgan/samples/toy'
                '/x_z_rec' + str(frame_index[0]) + '.png')
    plt.close()

    _x_, _inv_ = session.run([fake_data, inv_fake],
                             feed_dict={input_noise: batch_noise})
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].scatter(batch_noise[:, 0], batch_noise[:, 1], c='green', marker='x')
    ax[0].set_title('latent z')
    ax[0].set_xlim([-2., 2.])
    ax[0].set_ylim([-2., 2.])

    ax[1].scatter(_x_[:, 0], _x_[:, 1], c='orange', marker='+')
    ax[1].set_title('generated x')
    ax[1].set_xlim([-2., 2.])
    ax[1].set_ylim([-2., 2.])

    ax[2].scatter(_inv_[:, 0], _inv_[:, 1], c='red', marker='x')
    ax[2].set_title('inverted z')
    ax[2].set_xlim([-2., 2.])
    ax[2].set_ylim([-2., 2.])
    fig.savefig('/home/zhengliz/Output/improved-wgan/samples/toy'
                '/z_x_inv' + str(frame_index[0]) + '.png')
    plt.close()

    frame_index[0] += 1


def generate_from_noise():
    z = np.random.normal(loc=0.0, scale=0.5, size=(BATCH_SIZE, ZDIM))

    x = session.run(fake_data, feed_dict={input_noise: z})

    z_inv = session.run(inv_real, feed_dict={real_data: x})

    x_rec = session.run(rec_real, feed_dict={real_data: x})

    fig, ax = plt.subplots(1, 4, figsize=(17.9, 4))


    if ZDIM == 2:
        ax[0].scatter(z[:, 0], z[:, 1], c='green', marker='x')
        ax[0].set_title('z')
    elif ZDIM == 1:
        ax[0].hist(z[:, 0], 50, normed=1, histtype='bar')
        # ax[0].scatter(z[:, 0], np.zeros((z[:, 0].shape[0],)), c='green', marker='x')
        ax[0].set_title('z')


    ax[1].scatter(x[:, 0], x[:, 1], c='orange', marker='+')
    ax[1].set_title('x')

    if ZDIM == 2:
        ax[2].scatter(z_inv[:, 0], z_inv[:, 1], c='red', marker='x')
        ax[2].set_title('inv z')
    elif ZDIM == 1:
        ax[2].hist(z_inv[:, 0], 50, normed=1, histtype='bar')
        # ax[2].scatter(z_inv[:, 0], np.zeros((z[:, 0].shape[0],)), c='green', marker='x')
        ax[2].set_title('z')


    ax[3].scatter(x_rec[:, 0], x_rec[:, 1], c='red', marker='x')
    ax[3].set_title('reconstructed x')

    # plt.axes().set_aspect('equal', 'datalim')

    fig.savefig('/home/zhengliz/Output/improved-wgan/samples/toy'
                '/z_x_invz_recx_' + str(frame_index[0]) + '.png')
    plt.close()

    pickle.dump([z, x, z_inv, x_rec],
                open('/home/zhengliz/Output/improved-wgan/samples/toy/' +
                     'z_x_invz_recx_{}.sav'.format(frame_index[0]), 'wb'))

    frame_index[0] += 1


def generate_from_images(x_org, y_org):


    z_inv = session.run(inv_real, feed_dict={real_data: x_org})

    x_rec = session.run(rec_real, feed_dict={real_data: x_org})

    pickle.dump([x_org, y_org, z_inv, x_rec],
                open('/home/zhengliz/Output/improved-wgan/samples/toy/' +
                     'x_y_z_recx_{}.sav'.format(frame_index[0]), 'wb'))

    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4))

    ax[0].scatter(x_org[y_org == 0][:, 0], x_org[y_org == 0][:, 1], c='r', marker='o')
    ax[0].scatter(x_org[y_org == 1][:, 0], x_org[y_org == 1][:, 1], c='g', marker='x')
    ax[0].set_title('original x')

    if ZDIM == 2:
        ax[1].scatter(z_inv[y_org == 0][:, 0], z_inv[y_org == 0][:, 1], c='r', marker='o')
        ax[1].scatter(z_inv[y_org == 1][:, 0], z_inv[y_org == 1][:, 1], c='g', marker='x')
        ax[1].set_title('inv z')
    elif ZDIM == 1:
        stk = np.hstack((z_inv[y_org == 0][:, 0].reshape(-1,1), z_inv[y_org == 1][:, 0].reshape(-1,1)))
        ax[1].hist(stk, 50, normed=1, histtype='bar', stacked=True, color=['r','g'])
        # ax[1].scatter(z_inv[y_org == 0][:, 0], np.zeros((z_inv[y_org == 0][:, 0].shape[0],)), c='r', marker='o')
        # ax[1].scatter(z_inv[y_org == 1][:, 0], np.zeros((z_inv[y_org == 1][:, 0].shape[0],)), c='g', marker='x')
        ax[1].set_title('inv z')

    ax[2].scatter(x_rec[y_org == 0][:, 0], x_rec[y_org == 0][:, 1], c='r', marker='o')
    ax[2].scatter(x_rec[y_org == 1][:, 0], x_rec[y_org == 1][:, 1], c='g', marker='x')
    ax[2].set_title('reconstructed x')

    fig.savefig('/home/zhengliz/Output/improved-wgan/samples/toy/'
                'x_z_recx_{}.png'.format(frame_index[0]))
    plt.close()

    frame_index[0] += 1




# Train loop!
with tf.Session() as session:
    saver = tf.train.Saver(max_to_keep=1000)

    session.run(tf.global_variables_initializer())

    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        batch_noise = np.random.normal(loc=0.0, scale=0.5, size=(BATCH_SIZE, ZDIM))
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op, feed_dict={input_noise: batch_noise})
        # Train critic
        for i in xrange(CRITIC_ITERS):
            xy = gen.next()
            _data = xy[0]
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data, input_noise: batch_noise}
            )
            _inv_cost, _ = session.run(
                [inv_cost, inv_train_op],
                feed_dict={real_data: _data, input_noise: batch_noise}
            )

        # Write logs and save samples
        lib.plot.plot('disc cost', _disc_cost)
        lib.plot.plot('inv cost', _inv_cost)
        if iteration % 100 == 99:
            lib.plot.flush()
            generate_from_noise()
            generate_from_images(xy[0], xy[1])
        if iteration % 10000 == 9999:
            saver.save(session, '/home/zhengliz/Output/improved-wgan/models/toy/model',
                       global_step=iteration)
        lib.plot.tick()
