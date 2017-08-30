# graph1 = tf.Graph()
# with graph1.as_default():
#     model1 = Model1(args1)
#
# graph2 = tf.Graph()
# with graph2.as_default():
#     model2 = Model2(args2)
#
# sess1 = tf.Session(graph=graph1)
# sess2 = tf.Session(graph=graph2)
#
# with sess1.as_default():
#     with graph1.as_default():
#         tf.global_variables_initializer().run()
#         model1_saver = tf.train.Saver(tf.global_variables())
#         model1_ckpt = tf.train.get_checkpoint_state(args1.save_dir)
#         model1_saver.restore(sess1, model1_ckpt.model_checkpoint_path)
#
# with sess2.as_default():
#     with graph2.as_default():
#         tf.global_variables_initializer().run()
#         model2_saver = tf.train.Saver(tf.global_variables())
#         model2_ckpt = tf.train.get_checkpoint_state(args2.save_dir)
#         model2_saver.restore(sess2, model2_ckpt.model_checkpoint_path)
#
# with sess1.as_default():
#     pass
#
# with sess2.as_default():
#     pass
#
# sess1.close()
# sess2.close()




import numpy as np
import tensorflow as tf


# import tflib
import tflib.cifar10
DATA_DIR = '/home/zhengliz/Data/cifar10'
BATCH_SIZE = 64
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

train_gen, dev_gen = tflib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
x_batch, y_batch = train_gen().next()



from cifar10_classifier import Cifar10Classifier
classifier = Cifar10Classifier()
classifier.model.load("classifier_cifar10_cnn.tfl")

# y_first = classifier.model.predict(x_batch.reshape((-1,32,32,3)).astype('float32'))
# y_first = np.argmax(y_first, axis=1)
# for i, j in zip(y_batch, y_first):
#     print("({},{})".format(labels[i], labels[j])),
# print("\n")




graph2 = tf.Graph()
with graph2.as_default():
    from cifar10_wgan import Cifar10WGAN
    cifar10Wgan = Cifar10WGAN()
    saver2 = tf.train.Saver(max_to_keep=1000)

sess2 = tf.Session(graph=graph2)
with sess2.as_default():
    with graph2.as_default():
        saver2 = tf.train.import_meta_graph(
            '/home/zhengliz/Output/improved-wgan/models/cifar10/model-199999.meta')
        saver2.restore(
            sess2, '/home/zhengliz/Output/improved-wgan/models/cifar10/model-199999')
        cifar10Wgan.images_int = x_batch.reshape((BATCH_SIZE, -1)).astype('int')

        x, x_tilde = cifar10Wgan.generate_from_fixed_images(sess2, 123)


# prediction of original images
y = classifier.model.predict(x.reshape((-1,32,32,3)).astype('float32'))
y = np.argmax(y, axis=1)
# prediction of reconstructed images
y_tilde = classifier.model.predict(x_tilde.reshape((-1,32,32,3)).astype('float32'))
y_tilde = np.argmax(y_tilde, axis=1)
count = 0
for i, j, k in zip(y_batch, y, y_tilde):
    # (truth, pred of origin, pred of reconstruction)
    print "({},{},{}) ".format(labels[i], labels[j], labels[k]),
    count += (j != k)
print "\n{} out of {} reconstructions changed classifier " \
      "prediction\n".format(count, BATCH_SIZE)



