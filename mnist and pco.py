import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np
import pyswarms as ps

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def create_model(hyperparams):
    model = Sequential()
    model.add(Conv2D(hyperparams[0], kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(hyperparams[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hyperparams[2]))
    model.add(Flatten())
    model.add(Dense(hyperparams[3], activation='relu'))
    model.add(Dropout(hyperparams[4]))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return model

def fitness(hyperparams):
    model = create_model(hyperparams)
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=12,
              verbose=0,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    return 1 - score[1]

# Define the bounds and constraints for the hyperparameters
bounds = (np.array([16, 32, 0.1, 128, 0.1]), np.array([64, 128, 0.5, 512, 0.5]))
constraints = (np.array([1, 1, 0.05, 32, 0.05]), np.array([5, 5, 0.5, 128, 0.5]))

# Initialize the optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=5, options={'c1': 0.5, 'c2': 0.5, 'w': 0.5})

# Run the optimization process
best_cost, best_pos = optimizer.optimize(fitness, iters=10, bounds=bounds, constraints=constraints)

# Print the best hyperparameters and the corresponding fitness value
print('Best hyperparameters:', best_pos)
print('Best fitness value:', best_cost)

# Create the final model with the best hyperparameters
final_model = create_model(best_pos)

# Train the final model
final_model.fit(x_train, y_train,
                batch_size=128,
                epochs=12,
                verbose=1,
                validation_data=(x_test, y_test))

# Evaluate the final model
score = final_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

