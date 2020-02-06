# https://www.tensorflow.org/tutorials/load_data/images
# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_fundamentals.htm


import tensorflow as tf

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from os import path
import pathlib
import random
import math

class ModelTypes:
    TYPE_MLP = "MLP"
    TYPE_CNN = "CNN"

class TunerTypes:
    TYPE_GENETIC = "GENETIC"

class ParamTuner:
    def __init__(self, model_type, tuner_type):
        self.model_type = model_type
        self.tuner_type = tuner_type
        self.population = []
        self.population_size = 0
        self.tournamet_size = 2
        self.fitness_history = []

    def load_images(self, directory, show=False):
        self.directory = directory
        self.full_directory = pathlib.Path(path.expanduser("~") + directory)
        self.class_names = np.array([item.name for item in self.full_directory.glob('*')])
        self.image_count = len(list(self.full_directory.glob('*/*.png')))
        self.image_height = self.get_class_example(self.class_names[0]).shape[0]
        self.image_width = self.get_class_example(self.class_names[0]).shape[1]
        print(self.image_height)
        print(self.image_width)

    def get_image_count(self):
        return self.image_count

    def get_class_names(self):
        return self.class_names

    def get_class_example(self, class_name):
        class_directory = list(self.full_directory.glob(class_name + "/*"))

        example_directory = class_directory[0]
        img = plt.imread(str(example_directory))
        return img
    
    def display_batch(self):
        print("")
        print("Confirm that the images are labeled correctly and then exit.", end="\n\n")
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_data_gen = image_generator.flow_from_directory(directory=str(self.full_directory),
                                                            batch_size=25,
                                                            target_size=(self.image_height, self.image_width),
                                                            classes = list(self.class_names))
        
        image_batch, label_batch = next(train_data_gen)
        plt.figure(figsize = (10,10))
        for n in range(25):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(image_batch[n])
            plt.title(self.class_names[label_batch[n]==1][0].title())
            plt.axis('off')

        plt.show()
        
    def initialize_population(self, population_size, tournamet_size=2, n_input=1, n_output=1, n_nodes=100,
                                n_epochs=100, batch_size=100, n_hidden_layers=0, dropout=1,
                                activation_in='tanh',loss_fcn='mse',optimizer='adam', activation_out='tanh',
                                output_layer=True):
        
        self.population_size = population_size
        self.tournamet_size = tournamet_size
        for i in range(0, population_size):
            self.population.append([
                random.randint(1,n_input),
                random.randint(1,n_output),
                random.randint(1,n_nodes),
                random.randint(1,n_epochs),
                random.randint(1,batch_size),
                random.randint(0,n_hidden_layers),
                random.randint(1,dropout),
                activation_in,
                activation_out,
                loss_fcn,
                optimizer,
                output_layer
            ])

    def get_population(self):
        return self.population

    def print_chromosome(self, chromosome):
        print("")
        print("---------------CURRENT CHROMOSOME--------------", end="\n\n")
        print(chromosome, end="\n\n")
        print("-----------------------------------------------", end="\n\n")
            
    def fit_MLP(self, chromosome):
        self.print_chromosome(chromosome)
        (_, _, n_nodes, n_epochs, n_batch, n_hidden_layers, dropout, act_in, act_out, loss_fcn, optimizer, output_layer) = chromosome
        n_steps_per_epoch = np.ceil(self.image_count/n_batch)
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_data_gen = image_generator.flow_from_directory(directory=str(self.full_directory),
                                                            batch_size=n_batch,
                                                            target_size=(self.image_height, self.image_width),
                                                            classes = list(self.class_names))

        model = Sequential()
        model.add(Flatten(input_shape=(self.image_height, self.image_width, 3)))

        model.add(Dense(n_nodes, activation=act_in))
        
        for i in range(n_hidden_layers):
            model.add(Dense(int(n_nodes/2), activation = act_in))
            model.add(Dropout(dropout))

        if output_layer:
            model.add(Dense(len(self.class_names), activation = act_out))

        model.compile(loss = loss_fcn, optimizer = optimizer)

        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=n_steps_per_epoch,
            epochs=n_epochs,
        )

        self.fitness_history.append((history.history['loss'][-1], chromosome))

    def run_tournament_selection(self):
        fittest_loss = math.inf
        fittest_chromosome = []
        for i in range(0, self.tournamet_size):
            random_index = random.randint(0, len(self.fitness_history)-1)
            if self.fitness_history[random_index][0] < fittest_loss:
                fittest_loss = self.fitness_history[random_index][0]
                fittest_chromosome = self.fitness_history[random_index][1]

        return fittest_chromosome
                
    
    def select_parents(self):
        for i in range(0, self.population_size):
            parents = []
            for j in range(0, 2):
                parents.append(self.run_tournament_selection())

            print("")
            print("---------------PARENTS SELECTED----------------", end="\n\n")
            print(parents, end="\n\n")
            print("-----------------------------------------------", end="\n\n")

    def run_tuner(self):
        if self.tuner_type == TunerTypes.TYPE_GENETIC:

            for chromosome in self.population:
                self.fit_MLP(chromosome)

            print(self.fitness_history)

            self.select_parents()