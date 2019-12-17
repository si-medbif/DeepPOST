from absl import flags
from absl import app
from joblib import Parallel, delayed
import os
import re
import tensorflow as tf

 
# Assign flags
FLAGS = flags.FLAGS
flags.DEFINE_string("source_dir", None, "Path to labeled directory of tile files(e.g. Path/to/<CA/Benign>). Not directly to tile files")
flags.DEFINE_string("out_dir", None, "Path for saving TFRecord files")
flags.DEFINE_integer("jobs", -1, "Number of threads for parallel processing (default = -1 which means all CPUs will be used)")

# Required flags
flags.mark_flag_as_required("source_dir")
flags.mark_flag_as_required("out_dir")

class Slide_Processor:
  """A class for sorting tile files into a slide group to be written as TFRecord """
  def __init__(self,source_dir):
  	self.labels = [name  for _, dirs, _ in os.walk(source_dir) for name in dirs]
  	self.raw_path = [(roots.replace(source_dir.rstrip("/")+"/",''),roots,name,re.sub(r'_\d+_\d+\.jpeg','',name)) for roots,_,files in os.walk(source_dir) for name in files]
  	self.slides = list(set([name for _,_,_,name in self.raw_path]))
  def tiles_for_slide(self,slide_name):
  	result = [(os.path.join(root,name),name,label) for label,root,name,unique in self.raw_path if unique == slide_name]
  	return result

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(filepath, filename, label, label_index_list):
  image_string = open(filepath, 'rb').read()
  image_shape = tf.image.decode_jpeg(image_string).shape
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'image/filename': _bytes_feature(filename.encode('utf-8')),
      'image/encoded':  _bytes_feature(image_string),
      'image/class/label': _int64_feature(label_index_list.index(label)+1),
      'image/class/text': _bytes_feature(label.encode('utf-8'))
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def TFRecord_writer(filename, out_dir, slide_processor):
  new_filename = os.path.join(out_dir,"%s.tfrecord" % filename)
  with tf.io.TFRecordWriter(new_filename) as writer:
  	label_index_list = slide_processor.labels
  	observations = slide_processor.tiles_for_slide(filename)
  	for observation in observations:
  		example = serialize_example(observation[0],observation[1],observation[2],label_index_list)
  		writer.write(example)
  	print("%s.tfrecord is completely written with %d tiles" % (filename,len(observations)))

  
def main(argv):
  del argv  # Unused.

  s1 = Slide_Processor(FLAGS.source_dir)
  print(s1.labels)
  #print(len(s1.raw_path))

  inputs = [(name,FLAGS.out_dir,s1) for name in s1.slides]

  Parallel(n_jobs=FLAGS.jobs)(delayed(TFRecord_writer)(filename, out_dir, slide_processor) for filename, out_dir, slide_processor in inputs)
  
if __name__ == '__main__':
  app.run(main)
