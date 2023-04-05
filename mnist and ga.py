import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape images to 28 x 28 x 1 (for grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert target variable to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define hyperparameter search space
search_space = {
    'conv1_filters': [16, 32, 64],
    'conv2_filters': [32, 64, 128],
    'kernel_size': [3, 5, 7],
    'dense1_neurons': [64, 128, 256],
    'dense2_neurons': [32, 64, 128]
}

# Define genetic algorithm parameters
population_size = 10
mutation_rate = 0.1
generations = 10

# Define fitness function
def fitness_function(individual):
    # Define model architecture
    model = Sequential()
    model.add(Conv2D(individual['conv1_filters'], kernel_size=(individual['kernel_size'], individual['kernel_size']),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(individual['conv2_filters'], kernel_size=(individual['kernel_size'], individual['kernel_size']), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(individual['dense1_neurons'], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(individual['dense2_neurons'], activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=5,
              verbose=0)

    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]

# Define genetic algorithm functions
def initialize_population(population_size, search_space):
    population = []
    for i in range(population_size):
        individual = {}
        for key in search_space:
            individual[key] = random.choice(search_space[key])
        population.append(individual)
    return population

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(individual, search_space, mutation_rate):
    mutated_individual = {}
    for key in individual:
        if random.random() < mutation_rate:
            mutated_individual[key] = random.choice(search_space[key])
        else:
            mutated_individual[key] = individual[key]
    return mutated_individual
def select_parents(population, fitness_scores):
    # Select parents using tournament selection
    parent1_idx = random.randint(0, len(population)-1)
    parent2_idx = random.randint(0, len(population)-1)
    while parent2_idx == parent1_idx:
        parent2_idx = random.randint(0, len(population)-1)
        if fitness_scores[parent1_idx] > fitness_scores[parent2_idx]:
            return population[parent1_idx], parent1_idx
        else:
            return population[parent2_idx], parent2_idx
# Call genetic algorithm functions
population = initialize_population(population_size, search_space)

for generation in range(generations):
    print(f"Generation {generation+1}/{generations}")

    # Evaluate fitness of population
    fitness_scores = [fitness_function(individual) for individual in population]

    # Select parents and generate offspring
    new_population = []
    for i in range(population_size):
        parent1, parent1_idx = select_parents(population, fitness_scores)
        parent2, parent2_idx = select_parents(population, fitness_scores)
        child = crossover(parent1, parent2)
        child = mutate(child, search_space, mutation_rate)
        new_population.append(child)

    # Set the new population
    population = new_population

    # Find and print best individual
    best_individual_idx = np.argmax(fitness_scores)
    best_individual = population[best_individual_idx]
    best_fitness_score = fitness_scores[best_individual_idx]
    print(f"Best individual: {best_individual}, Fitness: {best_fitness_score}")

 