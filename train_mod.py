import os
import json
from absl import flags
from absl import app
from utils.dataset import get_dataset, count_dataset
from utils.build_model import build_model, build_model_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from datetime import datetime
import tensorflow as tf

#Allow GPU memory growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Assign flags
FLAGS = flags.FLAGS
#Required flags
flags.DEFINE_string("dataset_dir", None, "Path to Train/Valid TFRecords")
flags.DEFINE_string("out_dir", None, "Path for saving TFRecord files")
flags.DEFINE_string("model", None, "1) Create a fresh model to call from keras.applications (e.g. InceptionResNetV2, InceptionV3) or 2) Path to model.json for transferred learning (Must specify saved weight)")

#Flag for transfer learning
flags.DEFINE_string("chkpt", None, "Path to saved weight checkpoint for transfer learning (Imagenet weights if not specified)")

#Hyperparameter flags
flags.DEFINE_float("learning_rate",0.001,"Specify learning rate (default = 0.001)")
flags.DEFINE_integer("batch_size",1,"Batch size (default = 1)")
flags.DEFINE_integer("epochs",40,"Number of epoch for training (default = 40)")
flags.DEFINE_boolean("fine_tuning", False, "Specify fine tuning strategy for training (i.e. only train the output layer)")

#Flags for building a fresh model. Not used for transfer learning.
flags.DEFINE_string("pooling", "avg", "Pooling strategy: max, min or avg (default = avg)")
flags.DEFINE_integer("image_size",299,"Specify one dimension (i.e. width or height) of training/validating images in pixels (default = 299 pixels). The images are expected to be square.")
flags.DEFINE_integer("label_length", 2, "Specify a number of all possible labels (default = 2 for positive and negative)")


# Required flags
flags.mark_flag_as_required("dataset_dir")
flags.mark_flag_as_required("out_dir")
flags.mark_flag_as_required("model")

def main(argv):
  del argv #Unused

  if "json" in FLAGS.model:
  #Create a model from saved structure (json file) and weights (ckpt file)
    model = build_model_json(model_json = FLAGS.model,
                             weight = FLAGS.chkpt,
                             fine_tuning = FLAGS.fine_tuning)
  else:
  #Create a fresh model or a transferred model with saved weights, but no json file for model structure
    model = build_model(FLAGS.model,
					  label_length=FLAGS.label_length,
					  pooling=FLAGS.pooling,
					  IMAGE_SIZE= (FLAGS.image_size,FLAGS.image_size),
					  weight=FLAGS.chkpt,
					  fine_tuning=FLAGS.fine_tuning)

  #Save structure of the model as json file to be used in evaluation and transfer learning with identical model structure
  model_json = model.to_json()
  with open("%s/model.json" % FLAGS.out_dir.rstrip("/"), 'w') as f:
    f.write(model_json)

  #Compile the model
  model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=FLAGS.learning_rate),
              metrics=['accuracy'])

  #Prepare datasets from TFRecord shards
  count_train = count_dataset(FLAGS.dataset_dir, 'train', FLAGS.batch_size,FLAGS.epochs)
  count_valid = count_dataset(FLAGS.dataset_dir, 'valid', FLAGS.batch_size,FLAGS.epochs)

  ds_train = get_dataset(FLAGS.dataset_dir, 'train', FLAGS.batch_size,FLAGS.epochs)
  ds_valid = get_dataset(FLAGS.dataset_dir, 'valid', FLAGS.batch_size,FLAGS.epochs)


  #Setup a checkpoint
  filepath="%s/weights-improvement-{epoch:03d}-{val_accuracy:.3f}.ckpt" % FLAGS.out_dir.rstrip("/")
  checkpoint = ModelCheckpoint(filepath,
							   monitor='val_accuracy',
							   verbose=1,
							   save_best_only=True,
							   save_weights_only = True,
							   mode='max')

  #Setup TensorBoard
  os.makedirs("%s/logs/scalars/" % FLAGS.out_dir.rstrip("/"), exist_ok=True)
  logdir = ("%s/logs/scalars/" % FLAGS.out_dir.rstrip("/")) + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = TensorBoard(log_dir=logdir,
									 histogram_freq=1)

  #Setup early stopping
  earlystop_callback = EarlyStopping(
      monitor='val_accuracy', min_delta=0.0001,
      patience=5)

  #Fit the model
  model.fit(
        x=ds_train,
        steps_per_epoch=count_train,
        validation_data=ds_valid,
        validation_steps=count_valid,
        callbacks=[checkpoint, tensorboard_callback, earlystop_callback],
        epochs=FLAGS.epochs)

if __name__ == '__main__':
  app.run(main)
