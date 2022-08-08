import shutil
import tensorflow as tf
import os
import re
import string

if not os.path.exists('aclImdb'):
    # Alternatively just use: 
    #:$ curl https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz | tar -xz
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                        untar=True, cache_dir='.',
                                        cache_subdir='')
    

dataset_dir = 'aclImdb'
train_dir = os.path.join(dataset_dir, 'train')
unsupervised_traindir = os.path.join(train_dir, 'unsup')

if os.path.exists(unsupervised_traindir):
    shutil.rmtree(unsupervised_traindir)

batch_size = 42
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  f'[{re.escape(string.punctuation)}]',
                                  '')

max_features = 10_000
sequence_length = 20

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize = custom_standardization,
    max_tokens = max_features,
    output_mode = 'int',
    output_sequence_length = sequence_length)

# https://www.tensorflow.org/tutorials/keras/text_classification#prepare_the_dataset_for_training
