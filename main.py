#from test_psnr import SDGAN   #test
#from lsgan_model_pro import SDGAN  #train
from vdsr_model import SDGAN  # 20layers
import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Number of epoch [10]")
flags.DEFINE_integer("max_iter_step", 1000000, "Number of iter_step [20000]")
flags.DEFINE_integer("LAMBDA", 10, "Number of hp")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 64, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 64, "The size of label to produce [21]")
#flags.DEFINE_integer("output_size", 64, "The size of output in the first conv[64]")
flags.DEFINE_float("learning_rate_ger", 0.0001, "The learning rate of gradient descent algorithm [5e-5]")
flags.DEFINE_float("learning_rate_dis", 0.0001, "The learning rate of gradient descent algorithm [5e-5]")
flags.DEFINE_float("clamp_lower", -0.01, "the upper bound and lower bound of parameters in discriminator")
flags.DEFINE_float("clamp_upper", 0.01, "the upper bound and lower bound of parameters in discriminator")
flags.DEFINE_float("hp_re", 0.0001, "hyper-parameters of generator loss")
flags.DEFINE_float("hp_tv", 0.005, "hyper-parameters of generator loss")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
#flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("Citers", 5, "update Citers times of disciminator in one iter(unless i < 25 or i % 500 == 0, i is iterstep)")
flags.DEFINE_integer("stride", 32, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("log_dir", "log", "directory to store log, including loss and grad_norm of generator and discriminator")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train",False, "True for training, False for testing [True]")
flags.DEFINE_integer("b", 1, "This parameter indicates that D is generated as a real class")
flags.DEFINE_integer("a", 0, "This parameter indicates that G is generated as a fake class")
flags.DEFINE_integer("c", 1, "This parameter indicates that G is generated as a real class")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)


    with tf.Session() as sess:
        sdgan = SDGAN(sess,
                      image_size=FLAGS.image_size,
                      label_size=FLAGS.label_size,
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir,
                      log_dir=FLAGS.log_dir)

        sdgan.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
