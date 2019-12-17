import os
import tensorflow as tf
import tensorflow.keras.applications as keras_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model, load_model, model_from_json
import efficientnet.tfkeras as efn

def select_model(model_name):
  # Select a model from keras.applications
  try:
      m = eval("keras_model.%s" % model_name)
      return m
  except:
      exit("Model name is invalid. Please select an available model in keras.applications" or "efficientnet.tfkeras")

def select_efn_model(model_name):
  # Select a model from keras.applications
  try:
      m = eval("efn.%s" % model_name)
      return m
  except:
      exit("Model name is invalid. Please select an available model in keras.applications" or "efficientnet.tfkeras")

def fine_tune(model, fine_tuning):
  if fine_tuning:
    for layer in model.layers[:-1]:
      layer.trainable = False
  else:
    for layer in model.layers:
      layer.trainable = True
  return model

def build_model_json(model_json,weight,fine_tuning = False):
  '''
  Build a keras model with
  1) Saved structure as json
  2) Saved weight as a checkpoint
  '''
  with open(model_json, 'r') as f:
    loaded_model_json = f.read()
  loaded_model = model_from_json(loaded_model_json)
  if weight != None and weight != "Random":
    loaded_model.load_weights(weight)
  loaded_model = fine_tune(loaded_model,fine_tuning)

  return loaded_model

def build_model(model_name, label_length = 2, pooling = "avg", IMAGE_SIZE = None, weight = None, fine_tuning = False):
  '''
  Build a model from keras.applications with
  1. A new output layer with label + 1 nodes for an additional "background" node
  2. Allow weight loading from saved checkpoint (default = imagenet weights" or "Random" for random initial weights)
  3. Allow fine tuning (i.e. keep all layers except the output layer untrained)
  '''
  if "Efficient" in model_name:
    base_model = select_efn_model(model_name)
  else:
    base_model = select_model(model_name)

  if weight == None:
    init_weight = "imagenet"
  else:
    init_weight = None

  if IMAGE_SIZE == None:
    base_model = base_model(include_top=False, pooling=pooling, weights=init_weight)
  else:
    base_model = base_model(include_top=False, pooling=pooling, weights=init_weight,
                             input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
  if "Efficient" in model_name:
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(label_length + 1, activation="softmax")(x)
    model = Model(base_model.input, output)
  else:
    output = Dense(label_length + 1, activation='softmax')(base_model.output)
    model = Model(base_model.input, output)

  if weight != None and weight != "Random":
    model.load_weights(weight)

  model = fine_tune(model,fine_tuning)

  return model
