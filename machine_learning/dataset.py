import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt	
from mpl_toolkits.mplot3d import Axes3D

#data
class lg_dataset:

    def __init__(self):
        self.train_size = 2000

        self.weights = tf.constant([[3], [5]], dtype=tf.float32)
        self.biases = tf.random.uniform(
            [self.train_size, 1], minval=-5, maxval=5)

        self.train_param1 = tf.random.uniform(
            [self.train_size], minval=2, maxval=10)
        self.train_param2 = tf.random.uniform(
            [self.train_size], minval=50, maxval=110)

        self.train_data = tf.stack(
            [self.train_param1, self.train_param2], axis=1)
        self.train_label = tf.linalg.matmul(
            self.train_data, self.weights)+self.biases

        self.test_size = 2000
        self.test_param1 = tf.random.uniform(
            [self.test_size], minval=200, maxval=1000)
        self.test_param2 = tf.random.uniform(
            [self.test_size], minval=500, maxval=1100)

        self.test_data = tf.stack([self.test_param1, self.test_param2], axis=1)
        self.test_label = tf.linalg.matmul(
            self.test_data, self.weights)+self.biases

    def load_data(self):
        return (self.train_data, self.train_label), (self.test_data, self.test_label)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('param1')
        ax.set_ylabel('param2')
        ax.set_zlabel('targets')
        ax.scatter(self.test_param1, self.test_param2,
                   self.test_label, c='g', label="ground truth")
        ax.legend()
        plt.show()


def load_data(dataset_name, plot=False):
    if dataset_name == 'linear_regression':
        __dataset = lg_dataset()
        if plot:
            __dataset.plot()
        return __dataset.load_data()
    else:
        raise ValueError('Dataset not found')


if __name__ == '__main__':
    (train_data, train_label), (test_data, test_label) = load_data(
        'linear_regression', plot=True)
    print("train_data shape {} test data shape {}".format(
        train_data.shape, test_data.shape))
    print(train_data)
    print(train_label, test_label)
