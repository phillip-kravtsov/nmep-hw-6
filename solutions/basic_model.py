import os, copy
import tensorflow as tf
from . import utils

class BasicTFModel(object):
    def __init__(self, config):
        if config.best:
            config.update(self.get_best_config())
        self.config = copy.deepcopy(config)
        self.random_seed = self.config.random_seed

        self.set_model_params()
        
        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=50,
            )

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        self.set_model_params()
        if config.train:
            self.make_train_graph()

    def set_model_params():
        raise Exception('The set model params function must be overriden by model')

    def get_best_config(self):
        return utils.attr_dict.AttrDict({})

    @staticmethod
    def get_random_config(fixed_params={}):
        raise Exception('The get_random_config fuction must be overriden by the model')
    def infer(self, graph):
        raise Exception("The infer function must be overriden by the model")

    def learn_on_epoch(self):
        raise Exception("Override the learn on epoch function")

    def train(self):
        for epoch_id in range(0, self.num_epochs):
            self.learn_from_epoch()

            if save_every > 0 and epoch_id % save_every == 0:
                self.save()

    def save(self):
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, episode_id = self.sess.run([global_step_t, self.episode_id])
        if self.config.debug:
            print("Saving to {} with global step {}".format(self.result_dir, global_step))
        
        self.saver.save(self.sess, self.result_dir + '/model-ep_' + str(episode_id), global_step)

