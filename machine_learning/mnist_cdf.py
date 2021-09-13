import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Classifier:
    def __init__(self, train_data, train_labels, test_data, test_labels, epochs=2000, lr=0.21,features = 500):
        train_data = tf.cast(train_data, dtype=tf.float32)/255.0
        self.train_data = tf.reshape(train_data, shape=(-1, 784))
        self.train_labels = tf.one_hot(
            train_labels, depth=10, dtype=tf.float32)

        test_data = tf.cast(test_data, dtype=tf.float32)/255.0
        self.test_data = tf.reshape(test_data, shape=(-1, 784))
        self.test_labels = tf.one_hot(test_labels, depth=10, dtype=tf.float32)

        self.weights = tf.Variable(tf.random.uniform(
            shape=[784, 10],maxval = 1,minval = 0, dtype=tf.float32))
        self.bias = tf.Variable(tf.random.uniform(
            shape=[10],maxval = 1,minval = 0, dtype=tf.float32))

        self.epochs = epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.perceptron = self.pca(self.train_data,components = features)

    def pca(self,data,components= 500):
        covar = tf.linalg.matmul(tf.transpose(data),data)
        e,v = np.linalg.eig(covar)
        data = tf.linalg.matmul(data,tf.transpose(v[:components]))
        return data

    @tf.function
    def model(self, data):
        linear = tf.linalg.matmul(data, self.weights) + self.bias

        mean = tf.math.reduce_mean(linear)
        std = tf.math.reduce_std(linear)
        x = (linear - mean) / (std * math.sqrt(2))
        return 0.5*(1 + tf.math.erf(x))

        
    def train(self,plot = True):
        training_loss = []
        testing_loss = []
        loss = lambda: self.loss(label,self.model(data))  #+ (0.1*(self.weights)**2)
        while self.epochs != 0 :
            data = self.train_data
            label = self.train_labels
            training_loss.append(np.mean(loss()))

            self.optimizer.minimize(loss, var_list=[
                                    self.weights, self.bias])

            if self.epochs % 10 == 0:
                (train_accuracy, test_accuracy) = self.testing()
                print('\r Training: {:.2f}% Testing: {:.2f}%  Epoch: {}'.format(
                    train_accuracy, test_accuracy, self.epochs), end=' ')
                sys.stdout.flush()


            data = self.test_data
            label = self.test_labels
            testing_loss.append(np.mean(loss()))

            self.epochs -= 1

        if plot:
            plt.plot(range(len(training_loss)),np.array(training_loss),c = 'r')
            plt.plot(range(len(testing_loss)),np.array(testing_loss),c = 'g')
            plt.show()
        print()

    def testing(self):
        train_predictions = np.argmax(self.model(self.train_data), axis=1)
        test_predictions = np.argmax(self.model(self.test_data), axis=1)

        train_truth = np.argmax(self.train_labels, axis=1)
        test_truth = np.argmax(self.test_labels, axis=1)

        count = 0
        for ind, elem in enumerate(train_predictions):
            if elem == train_truth[ind]:
                count += 1

        train_accuracy = (count/60000)*100

        count = 0
        for ind, elem in enumerate(test_predictions):
            if elem == test_truth[ind]:
                count += 1

        test_accuracy = (count/10000)*100

        return (train_accuracy, test_accuracy)


if __name__ == "__main__":
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.mnist.load_data()

    classify = Classifier(train_data, train_labels, test_data, test_labels)
    classify.train()
    classify.testing()
