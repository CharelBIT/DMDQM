import os
import pprint
import inspect
import json

import tensorflow as tf

pp = pprint.PrettyPrinter().pprint

def class_vars(obj):
  return {k:v for k, v in inspect.getmembers(obj)
      if not k.startswith('__') and not callable(k)}

class BaseModel(object):
  """Abstract object representing an Reader model."""
  def __init__(self, config):
    self._saver = None
    self.config = config

    try:
      print(config.__dict__['__flags'])
      self._attrs = config.__dict__['__flags']
    except:
      self._attrs = class_vars(config)
    pp(self._attrs)

    self.config = config

    for attr in self._attrs:
      name = attr if not attr.startswith('_') else attr[1:]
      setattr(self, name, getattr(self.config, attr))

  def save_model(self, step=None):
    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__
    print(model_name)

    if not os.path.exists(self.checkpoint_dir(step)):
      os.makedirs(self.checkpoint_dir(step))
    self.saver.save(self.sess, self.checkpoint_dir(step)+'/model.ckpt', global_step=step)

  def load_model(self,step):
    print(" [*] Loading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir(step))
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(self.checkpoint_dir(step), ckpt_name)
      self.saver.restore(self.sess, fname)
      print(" [*] Load SUCCESS: %s" % fname)
      return True
    else:
      print(" [!] Load FAILED: %s" % self.checkpoint_dir)
      return False


  def checkpoint_dir(self,step):
    return os.path.join('checkpoints',str(step) )



  def model_dir(self):
      save_json=json.dumps(self._attrs)
      with open(os.path.join(self.save_dir,'config.json'),'w') as f:
          f.write(save_json)
          f.close()
      return self.config.save_dir+'/'

  @property
  def saver(self):
    if self._saver == None:
      self._saver = tf.train.Saver(max_to_keep=10)
    return self._saver
