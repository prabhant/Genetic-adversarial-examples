
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import numpy as np
import matplotlib.pyplot as plt
import random as rand

# Ignore all GPUs, tf random forest does not benefit from it.
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# Parameters
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
print("predicted", )
'''''''''''
#====================================================================================================================#
#Trying out the similar approach as NN

mutation_degree = 0.5                              # Mutation probability
mutation_number = int(mutation_degree * 784)              # Number of elements to mutate
mutation_range = range(784)                        # Used to select mutation and crossover points
crossover_probability = 0.6                           # Crossover probability
population_number = 50                           # Population size
population_range = range(population_number)
generation_number = 1000                         # Number of generations
generation_range = range(generation_number)
tournaments = 3                            # Tournament sizes
guassian_value = 0.007                         # Determines magnitude of mutations (guassian standard deviation)
fitness_sensativity = 0.0000005             # Used in fitness algorithm, multiplies the distance between 2 images

sess = tf.InteractiveSession()

# Note: the following are placeholders for a tensor that will be constantly fed
x = tf.placeholder(tf.float32, shape=[None, 784])   # Feature tensor = 28x28

W = tf.Variable(tf.zeros([784, 10]))                # Weights
b = tf.Variable(tf.zeros([10]))                     # Biases

sess.run(tf.global_variables_initializer())         # Initializes TensorFlow variables and model saver
saver = tf.train.Saver()

batch = mnist.test.next_batch(1)                    # I noticed the second MNIST image is a 2 so I just hard coded it
batch = mnist.test.next_batch(1)
two = batch[0]                                      # two is our test image
population = np.zeros((population_number, 784))
for i in population_range:                              # Seeding the initial image population as the target image (two)
    population[i] = two

test = population[1]                                # Test sample from population

# Print the image
test.shape = (28, 28)
plt.imshow(test, cmap='gray')
plt.savefig("fig1.png")

children = np.zeros((population_number, 784))            # Empty child population


# Mutation function (with some probability, each pixel mutates in an amount given by guassian distribution)
def generate_mutation(chromosome):
    change_list = rand.sample(mutation_range, mutation_number)

    for i in change_list:
        chromosome[i] += np.random.normal(scale=guassian_value)    # The results are somewhat sensitive to how
                                                                # the gaussian is scaled
    chromosome[chromosome < 0] = 0
    chromosome[chromosome > 1] = 1

    return chromosome


# Crossover function (two point crossover)
def crossover(chromosome1, chromosome2):
    crossover_points = rand.sample(mutation_range, 2)

    temp = chromosome2[crossover_points[0]:crossover_points[1]]
    chromosome2[crossover_points[0]:crossover_points[1]] = chromosome1[crossover_points[0]:crossover_points[1]]
    chromosome1[crossover_points[0]:crossover_points[1]] = temp

    return chromosome1, chromosome2


# Tournament function (winner moves on with 100% probability)
def tournament(images, target_image, scores, target_score, step):
    fitness_value = np.zeros(tournaments)
    for i in range(tournaments):
        fitness_value[i] = fitness(images[i], target_image, scores[i], target_score, step)

    winner = images[np.argmax(fitness_value)]

    return winner


# Fitness function
def fitness(image, target_image, score, target_score, step):
    fitness_value = -(fitness_sensativity * np.linalg.norm(image - target_image) + (target_score-score))
    # The above equation is actually super sensitive to the first part, maybe worth reformulating somehow

    # If you zero out the second part, all the mutations are suppressed and zeroing out the first part causes
    # fast convergence to >99% certainty. But the balance needs to be tuned.

    return fitness_value


# Model from Tutorial 1
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Restore variables
saver.restore(sess, 'linear_model/linear_model.ckpt')

# Prints the initial classification
result = sess.run(y, feed_dict={x: batch[0]})
print("Model out:", result)
print("Actual:", batch[1])
two.shape = (784)

# Main algorithm
for i in generation_range:
    scores = sess.run(y, feed_dict={x: population})     # Score the entire population = target class % confidence

    # In each generation, we hold population/2 number of tournaments, 2 at a time, each with a randomly selected
    # sample from the population (with replacement). 60% of the time, the tournament winners crossbreed and their
    # children move on to the next generation, 40% of the time the tournament winners move on the next generation.
    # Every individual that moves on is mutated.
    for j in range(population_number//2):
        selection = rand.sample(population_range, tournaments)
        parent1 = tournament([population[selection[0]], population[selection[1]], population[selection[2]]], two,
                             [scores[selection[0], 6], scores[selection[1], 6], scores[selection[2], 6]], 1, i)
        selection = rand.sample(population_range, tournaments)
        parent2 = tournament([population[selection[0]], population[selection[1]], population[selection[2]]], two,
                             [scores[selection[0], 6], scores[selection[1], 6], scores[selection[2], 6]], 1, i)

        if np.random.rand() < crossover_probability:
            child1, child2 = crossover(parent1, parent2)
            children[j*2] = generate_mutation(child1)
            children[j*2+1] = generate_mutation(child2)
        else:
            children[j*2] = generate_mutation(parent1)
            children[j*2+1] = generate_mutation(parent2)

    population = children

# Take a sample from out final population, (I just took the first since they all should have converged to be similar)
# and checks the classification
test = population[1]
test.shape = (1, 784)
result = sess.run(y, feed_dict={x: test})
result_duplicate=result
print("result duplicate:",result_duplicate)
i_max=np.argmax(result)
print(i_max)            #finding the max probability
print(result.astype)
print(result.flatten())#flttening the numpy array from nd to 1D
result_list = result.flatten()  #saving array in another list
for j in range(len(result_list)):
    if result_list[i_max]==result_list[j]:
        result_list[j]=1
    else:
        result_list[j]=0        #making other prediction=0 for clearer results
print("result duplicate:",result_duplicate)
print ("New: ", result_list)

test.shape = (28, 28)
plt.imshow(test, cmap='gray')
plt.savefig("fig2.png")
''''''''''
