import os
import json
import tensorflow as tf
import gc
from utils.dataset import get_one_slide
from utils.build_model import build_model_json
from absl import flags
from absl import app

#Allow GPU memory growth
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)

#Allow GPU growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
    # Currently, memory growth needs to be the same across GPUs
#    for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
#    print(e)

#Do not allow GPU growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
  # Restrict TensorFlow to only use the first GPU
#  try:
#    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
#    print(e)

#Limit GPU memory at 2GB
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*2)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# Assign flags
FLAGS = flags.FLAGS
flags.DEFINE_string("source_dir", None, "Path to TFRecords of slide tile files")
flags.DEFINE_string("model_json", None, "Path to json file containing the structure of the model to be tested")
flags.DEFINE_string("chkpt", None, "Path to saved weight checkpoint for transfer learning (Imagenet weights if not specified)")
flags.DEFINE_string("out_dir", None, "Path for saving TFRecord files")
flags.DEFINE_integer("batch_size",1,"Batch size (default = 1)")


# Required flags
flags.mark_flag_as_required("source_dir")
flags.mark_flag_as_required("out_dir")
flags.mark_flag_as_required("model_json")

def test_one_slide(source_dir, file, out_dir, model, batch_size):
  print("Processing: %s" % file)

  out_file = file.replace("tfrecord","txt")
  with open(os.path.join(out_dir,out_file), 'w') as f:
    slide_set = get_one_slide(source_dir,file,batch_size)
    counter = 0
    for batch in slide_set:
      image,label,name = batch
      counter += name.shape[0]
      prediction = model.predict(image)
      res =  [name.numpy(),label.numpy(),prediction]
      for x in zip(*res):
        f.write("{0}\t{1}\t{2}\n".format(*x))
      gc.collect()
    print("%s Tiles has been processed" % counter)
  #tf.keras.backend.clear_session()
  #tf.compat.v1.reset_default_graph()import os
import json
import tensorflow as tf
import gc
from utils.dataset import get_one_slide
from utils.build_model import build_model_json
from absl import flags
from absl import app

#Allow GPU memory growth
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)

#Allow GPU growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
    # Currently, memory growth needs to be the same across GPUs
#    for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
#    print(e)

#Do not allow GPU growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
  # Restrict TensorFlow to only use the first GPU
#  try:
#    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
#    print(e)

#Limit GPU memory at 2GB
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*2)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# Assign flags
FLAGS = flags.FLAGS
flags.DEFINE_string("source_dir", None, "Path to TFRecords of slide tile files")
flags.DEFINE_string("model_json", None, "Path to json file containing the structure of the model to be tested")
flags.DEFINE_string("chkpt", None, "Path to saved weight checkpoint for transfer learning (Imagenet weights if not specified)")
flags.DEFINE_string("out_dir", None, "Path for saving TFRecord files")
flags.DEFINE_integer("batch_size",1,"Batch size (default = 1)")


# Required flags
flags.mark_flag_as_required("source_dir")
flags.mark_flag_as_required("out_dir")
flags.mark_flag_as_required("model_json")

def test_one_slide(source_dir, file, out_dir, model, batch_size):
  print("Processing: %s" % file)

  out_file = file.replace("tfrecord","txt")
  with open(os.path.join(out_dir,out_file), 'w') as f:
    slide_set = get_one_slide(source_dir,file,batch_size)
    counter = 0
    for batch in slide_set:
      image,label,name = batch
      counter += name.shape[0]
      prediction = model.predict(image)
      res =  [name.numpy(),label.numpy(),prediction]
      for x in zip(*res):
        f.write("{0}\t{1}\t{2}\n".format(*x))
      gc.collect()
    print("%s Tiles has been processed" % counter)
  
def main(argv):
  del argv #Unused

  model = build_model_json(model_json=FLAGS.model_json,
                           weight=FLAGS.chkpt)
  files = os.listdir(FLAGS.source_dir)

  for file in files:
    test_one_slide(FLAGS.source_dir,
                   file,
                   FLAGS.out_dir,
                   model,
                   FLAGS.batch_size)

if __name__ == '__main__':
  app.run(main)


def main(argv):
  del argv #Unused

  model = build_model_json(model_json=FLAGS.model_json,
                           weight=FLAGS.chkpt)
  files = os.listdir(FLAGS.source_dir)

  for file in files:
    test_one_slide(FLAGS.source_dir,
                   file,
                   FLAGS.out_dir,
                   model,
                   FLAGS.batch_size)

if __name__ == '__main__':
  app.run(main)
