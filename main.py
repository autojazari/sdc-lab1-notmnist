#%matplotlib inline

# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.variables import Variable

from tqdm import tqdm
import matplotlib.pyplot as plt

import prep


def unit_tests():
      assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
      assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
      assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
      assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

      #assert features._shape == (None) or features._shape == (None, 784), 'The of features is incorrect'
      #assert labels._shape == (None) or features._shape == (10), 'The shape of labels is incorrect'
      assert weights._variable._shape == (784, 10), 'The shape of weights is incorrect'
      assert biases._variable._shape == (10), 'The shape of biases is incorrect'
      
      assert features._dtype == tf.float32, 'features must be type float32'
      assert labels._dtype == tf.float32, 'labels must be type float32'


def setup_tensors():
      # Problem 2 - Set the features and labels tensors
      features = tf.placeholder(tf.float32, shape=(None, 784))
      labels = tf.placeholder(tf.float32, shape=(None))

      # Problem 2 - Set the weights and biases tensors
      weights = tf.Variable(tf.zeros((784, 10)))
      biases = tf.Variable(tf.zeros((10)))

       # Linear Regression Function WX + b
      logits = tf.matmul(features, weights) + biases

      prediction = tf.nn.softmax(logits)

      # Cross entropy
      cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

      # Training loss
      loss = tf.reduce_mean(cross_entropy)

      # Create an operation that initializes all variables
      init = tf.initialize_all_variables()
      return init, prediction, loss, features, labels


def load_features(features, labels):      
      # Reload the data
      pickle_file = 'notMNIST.pickle'
      with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
            train_features = pickle_data['train_dataset']
            train_labels = pickle_data['train_labels']
            valid_features = pickle_data['valid_dataset']
            valid_labels = pickle_data['valid_labels']
            test_features = pickle_data['test_dataset']
            test_labels = pickle_data['test_labels']
            del pickle_data  # Free up memoy

            print('Data and modules loaded.')

      # Feed dicts for training, validation, and test session
      train_feed_dict = {features: train_features, labels: train_labels}
      valid_feed_dict = {features: valid_features, labels: valid_labels}
      test_feed_dict = {features: test_features, labels: test_labels}

      return (train_features,
              train_labels,
              train_feed_dict,
              valid_feed_dict,
              test_feed_dict)

def test_biases(init):
      # Test Cases
      init  = setup_tensors()
      with tf.Session() as session:
            session.run(init)
            session.run(loss, feed_dict=train_feed_dict)
            session.run(loss, feed_dict=valid_feed_dict)
            session.run(loss, feed_dict=test_feed_dict)
            biases_data = session.run(biases)

      assert not np.count_nonzero(biases_data), 'biases must be zeros'

      print('Tests Passed!')


def setup_accuracy_function(prediction, labels):
      # Determine if the predictions are correct
      is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

      # Calculate the accuracy of the predictions
      accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
      print('Accuracy function created.')

      return accuracy


def train_model(session, args_dict):
      try:
            log_batch_step = args_dict['log_batch_step']
            batch_count = args_dict['batch_count']
            batch_size = args_dict['batch_size']
            epoch_i = args_dict['epoch_i']
            epochs = args_dict['epochs']
            accuracy = args_dict['accuracy']
            train_feed_dict = args_dict['train_feed_dict']
            valid_feed_dict = args_dict['valid_feed_dict']
            train_features = args_dict['train_features']
            train_labels = args_dict['train_labels']
            optimizer = args_dict['optimizer']
            loss = args_dict['loss']
            features = args_dict['features']
            labels = args_dict['labels']
            
      except KeyError as ke:
            print("Missing argument " + e)
            import sys;sys.exit(1)

      # Progress bar
      batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
      
      # The training cycle
      for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)


def test_model(session, accuracy, test_feed_dict):
      # Check accuracy against Test data
      test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)
      print('Nice Job! Test Accuracy is {}'.format(test_accuracy))


def plot_model(batches, loss_batch, train_acc_batch, valid_acc_batch):
      loss_plot = plt.subplot(211)
      loss_plot.set_title('Loss')
      loss_plot.plot(batches, loss_batch, 'g')
      loss_plot.set_xlim([batches[0], batches[-1]])
      acc_plot = plt.subplot(212)
      acc_plot.set_title('Accuracy')
      acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
      acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
      acc_plot.set_ylim([0, 1.0])
      acc_plot.set_xlim([batches[0], batches[-1]])
      acc_plot.legend(loc=4)
      plt.tight_layout()
      plt.ion()
      plt.show(block=True)
      #plt.show()


def main():
      # run the prep module's main which downloads and preprocesses
      # notMNIST's data set and creates pickle for future executions
      prep.prep_notmnist()
      init, prediction, loss,features, labels = setup_tensors()
      # Problem 3 - Tune the learning rate, number of epochs, and batch size to get a good accuracy
      learning_rate = 0.05
      epochs = 50

      (train_features,
      train_labels,
      train_feed_dict,
      valid_feed_dict,
      test_feed_dict) = load_features(features, labels)
      
      accuracy = setup_accuracy_function(prediction, labels)
      
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
      batch_size = int(len(train_features) * 0.009)
      print("batch size", batch_size)
      
      # Measurements use for graphing loss and accuracy
      log_batch_step = 50
      batches = []
      loss_batch = []
      train_acc_batch = []
      valid_acc_batch = []

      with tf.Session() as session:
            session.run(init)
            batch_count = int(math.ceil(len(train_features)/batch_size))
            
            for epoch_i in range(epochs):
                  args_dict = dict(
                        log_batch_step = log_batch_step,
                        batch_count = batch_count,
                        batch_size = batch_size,
                        epoch_i = epoch_i,
                        epochs = epochs,
                        accuracy = accuracy,
                        train_feed_dict = train_feed_dict,
                        valid_feed_dict = valid_feed_dict,
                        train_features = train_features,
                        train_labels = train_labels,
                        optimizer = optimizer,
                        loss = loss,
                        labels = labels,
                        features = features
            
                  )

            train_model(session, args_dict)
            test_model(session. accuracy, test_feed_dict)
      
      plot_model(batches, loss_batch, train_acc_batch, valid_acc_batch)

# execute!
main()
