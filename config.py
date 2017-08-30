class AgentConfig(object):
  #image_dir=''
  buffer_size=1000000
  explore_iter=10000
  is_explore=True
  explore_train_iter=10000
  explore_test_iter=100
  save_dir=''
  dataset_dir=''
  model_path=''
  action_overturn=False
  
  test_interval=-1
  test_iter=100
  camera_id=4
  
  scale = 10000
  #train_action=''
  camera_id=1
  save_iter=10*scale
  coor_action_size=27
  radian_action_size=27
  
  display = False

  max_step = 500*scale


  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.90
  target_q_update_step = 1 * scale
  learning_rate = 0.00025
  learning_rate_minimum = 0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 1 * scale
  
  is_train=True

  ep_end = 0.1
  ep_start = 1.
  #ep_end_t = memory_size

  history_length = 4
  train_frequency = 4
  learn_start = 0

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False
  radian_coor_coef=0.5

  _snap_step = 1000
  _save_step = 1000

class EnvironmentConfig(object):
  env_name = 'ARM'

  screen_width  = 200
  screen_height = 200
  max_reward = 1.
  min_reward = 0.
  camera_id=1
  HOST=""
  PORT=123
  coor_move_step=0.002
  radian_move_step=0.001

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)
  print(config)
  return config
