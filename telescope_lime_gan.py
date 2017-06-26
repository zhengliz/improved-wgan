import os
import sys

import matplotlib

matplotlib.use('Agg')
sys.path.append(os.getcwd())
sys.path.insert(1, '../lime')

import lime
from lime.lime_tabular import GanTabularExplainer
from gan_telescope_inv import *

from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

  # train a RF classifier
  classifier = RandomForestClassifier(n_estimators=10)
  classifier.fit(X_dev, y_dev)

  # load WGAN model
  saver = tf.train.Saver(max_to_keep=1000)

  with tf.Session() as session:
    # saver = tf.train.import_meta_graph(os.path.join(
    #   OUTPUT_PATH, 'telescope_to_200000_vary_sigma/model-199999.meta'))
    # saver.restore(session, os.path.join(
    #   OUTPUT_PATH, 'telescope_to_200000_vary_sigma/model-199999'))

    # train the Invertor
    # for iteration in xrange(10000):
    #   _input_noise = np.random.normal(size=(BATCH_SIZE, NOISE_DIM))
    #   _inv_cost, _ = session.run([inv_cost, inv_train_op],
    #                              feed_dict={input_noise: _input_noise})
    #   lib.plot.plot('telescope train invertor cost', _inv_cost)
    #   if iteration % 100 == 99:
    #     lib.plot.flush()
    #   lib.plot.tick()
    #
    # saver.save(session, os.path.join(
    #   OUTPUT_PATH, "models/telescope/model_invertor"), global_step=iteration)

    saver = tf.train.import_meta_graph(os.path.join(
      OUTPUT_PATH, 'models/telescope/model_invertor-9999.meta'))
    saver.restore(session, os.path.join(
      OUTPUT_PATH, 'models/telescope/model_invertor-9999'))

    feature_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
                     'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
    gan_explainer = GanTabularExplainer(training_data=X_dev,
                                        training_labels=y_dev,
                                        feature_names=feature_names,
                                        verbose=True,
                                        class_names=['signal', 'background'])
    for j in xrange(10):
      data_row = np.reshape(X_dev[j], (-1, 10))
      perturbed_samples = sample_generator(session, data_row, 5000)
      gan_explainer.set_gan_samples(perturbed_samples, perturbed_samples)
      ret_exp = gan_explainer.explain_instance(
        data_row.ravel(), classifier.predict_proba, num_features=5)
      ret_exp.save_to_file(os.path.join(OUTPUT_PATH,
                                        'explanation_{}.html'.format(j)))
