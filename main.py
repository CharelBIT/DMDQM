from __future__ import print_function
import random
import tensorflow as tf
from DQN.agent import Agent
from config import get_config
from DQN.environment import ARMEnv
flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', True, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', True, 'Whether to use double q-learning')

flags.DEFINE_integer('action_repeat', 1, 'The number of action to be repeated')
flags.DEFINE_integer('screen_width',84,'')
flags.DEFINE_integer('screen_height',84,'')
flags.DEFINE_integer('camera_num',4,'')
flags.DEFINE_integer('buffer_size',1000000,'')
flags.DEFINE_bool('is_exploring',True,'')
flags.DEFINE_integer('explore_iter',10000)
flags.DEFINE_integer('explore_train_iter',10000,'')
flags.DEFINE_integer('explore_test_iter',100,'')
# Etc
flags.DEFINE_boolean('use_gpu',True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', True, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train',True,'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_string('model_path', 'net/model.ckpt-50000', 'model dir')
flags.DEFINE_string('dataset_dir', '/home/officer/data/dataset_sample', 'model dir')

flags.DEFINE_string('HOST',"192.168.31.52",'Enviroment')
flags.DEFINE_integer('PORT',30003,'Enviroment')

flags.DEFINE_integer('coor_action_size', 27, 'robot depth of free')
flags.DEFINE_integer('radian_action_size', 27, 'robot depth of free')
#flags.DEFINE_integer('save_iter', 1000, 'save model per snapshot_iter')
flags.DEFINE_integer('camera_id', 0, 'robot depth of free')
flags.DEFINE_float('coor_move_step', 0.001, '')
flags.DEFINE_float('radian_move_step', 0.001, '')


flags.DEFINE_integer('test_interval',100,'')
flags.DEFINE_integer('test_iter',100,'')
flags.DEFINE_boolean('action_overturn',True,'')


FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")
  
print(FLAGS.__dict__)

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS
    if config.env_name=='ARM':
        env=ARMEnv(config)
    else:
        raise Exception("NoElem")

    if not tf.test.is_gpu_available() and FLAGS.use_gpu:
      raise Exception("use_gpu flag is true when no GPUs are available")

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    agent = Agent(config, env, sess)

    if FLAGS.is_train:
      agent.train()
    elif FLAGS.is_exploring:
      agent.explore()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
