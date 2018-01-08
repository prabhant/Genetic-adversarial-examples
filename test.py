from mnist_reader import load_mnist
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant

# Load data
# Function load_minst is available in git.
X_train, y_train = load_mnist('input/data', kind='train')
X_test, y_test = load_mnist('input/data', kind='t10k')
data_predict = load_mnist('input/data')

# Prepare datasets
# This step contains normalization and reshaping of input.
# For output, it is important to change number to one-hot vector.
X_train = X_train.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Create model in Keras
# This model is linear stack of layers
clf = Sequential()

# This layer is used as an entry point into a graph.
# So, it is important to define input_shape.
clf.add(
    InputLayer(input_shape=(1, 28, 28))
)

# Normalize the activations of the previous layer at each batch.
clf.add(
    BatchNormalization()
)

# Next step is to add convolution layer to model.
clf.add(
    Conv2D(
        32, (2, 2),
        padding='same',
        bias_initializer=Constant(0.01),
        kernel_initializer='random_uniform'
    )
)

# Add max pooling layer for 2D data.
clf.add(MaxPool2D(padding='same'))

# Add this same two layers to model.
clf.add(
    Conv2D(
        32,
        (2, 2),
        padding='same',
        bias_initializer=Constant(0.01),
        kernel_initializer='random_uniform',
        input_shape=(1, 28, 28)
    )
)



clf.add(MaxPool2D(padding='same'))

# It is necessary to flatten input data to a vector.
clf.add(Flatten())

# Last step is creation of fully-connected layers.
clf.add(
    Dense(
        128,
        activation='relu',
        bias_initializer=Constant(0.01),
        kernel_initializer='random_uniform',
    )
)

# Add output layer, which contains ten numbers.
# Each number represents cloth type.
clf.add(Dense(10, activation='softmax'))

# Last step in Keras is to compile model.
clf.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

clf.predict()