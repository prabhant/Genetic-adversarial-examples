# Just a test script to see the trained model's confidence levels
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # This is our training data

import tensorflow as tf
sess = tf.InteractiveSession()

# Note: the following are placeholders for a tensor that will be constantly fed
x = tf.placeholder(tf.float32, shape=[None, 784])   # Feature tensor = 28x28

W = tf.Variable(tf.zeros([784, 10]))    # Weights
b = tf.Variable(tf.zeros([10]))         # Biases

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# Model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Restore variables
saver.restore(sess, 'linear_model/linear_model.ckpt')

# Outputs y and y_ for first 100 test images
for i in range(100):
    batch = mnist.test.next_batch(1)
    result = sess.run(y, feed_dict={x: batch[0]})
    print("Model out:", result)
    print("Actual:", batch[1])

sess.close()