import random as rand
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)   # This is the MNIST data
#=====================================================================================================================
sess = tf.InteractiveSession()

# In[3]:


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# In[4]:


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# In[6]:


sess.run(tf.global_variables_initializer())
y = tf.matmul(x, W) + b

# In[9]:


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# In[11]:


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[12]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[13]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# In[15]:


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# In[16]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# In[17]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# In[18]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# In[19]:


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# In[23]:


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    save_path = saver.save(sess, "fashion_model/fashion_model.ckpt")






#======================================================================================================================
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

# Print image
test.shape = (28, 28)
plt.imshow(test, cmap='gray')
plt.savefig("fig2.png")
