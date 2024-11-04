import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, History
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from kerastuner.tuners import RandomSearch
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt


class NeuralNet(object):
    
    def __init__(self, checkpoint_path, input_shape, train_data, val_data, nodes = [40, 10 ,1], model = None, epochs = 15, batch_size = 128):
        '''
        The initialization which creates our network
        
        nodes: The number of nodes for each layer
        model: The optional model to create the net with
        '''
        self.history = None
        # The training data
        self.train_data = train_data
        # The validation data
        self.val_data = val_data
        # The number of nodes (unused with hpyerparameter tuning)
        self.nodes = nodes
        # The batch size
        self.batch_size = batch_size
        # The number of epochs
        self.epochs = epochs
        # The shape of the image
        self.input_shape = input_shape
        # The callback to store history
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,\
                                           save_weights_only=True, \
                                           verbose=0, save_freq=5)
    
        '''
        Creates a model if none is given
        '''
        if model is None:
            # A tuner object which searches for the best parameters
            self.tuner = RandomSearch(self.create_model, objective = \
                                      "val_accuracy", max_trials = 3, \
                                      executions_per_trial = 1)
            # Rpeatedly creates and checks the model to determine the correct
            # parameters
            self.tuner.search(x=self.train_data, epochs = 30, batch_size = 64,\
                              validation_data = self.val_data, verbose = 0)
            # Creates the model based on the best parameters as determined by
            # the search
            self.model = \
            self.create_model(\
                              self.tuner.get_best_hyperparameters(\
                                                                  num_trials=\
                                                                  1)[0])
        else:
            # Takes a pre-created model if provided one in the constructor
            self.model = model
        
    def create_model(self, hp):
        '''
        Creates the model in Keras if none was provided
        '''
        model = Sequential()
        # Takes the multi-dimensional image data and makes it readable by the
        # model
        model.add(Flatten(input_shape = (self.input_shape[0], \
                                         self.input_shape[1], \
                                         self.input_shape[2])))
        # Creates the correct number of layers and nodes based on the search
        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(Dense(hp.Int(f"Dense_layer_{i}_units", 32, 512, 32), \
                            activation=hp.Choice(
                    f"dense_layer_{i}_activation",
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'), kernel_regularizer=l2(\
                                  hp.Choice("dense_layer_{i}_regularization", \
                                            values=[1e-3, 1e-4, 1e-5, 1e-6])),\
                                    bias_regularizer=l2(\
                                    hp.Choice(\
                                    "dense_layer_{i}_bias_regularization", \
                                    values=[1e-3, 1e-4, 1e-5, 1e-6]))))
        # Dropout helps to reduce overfitting     
        model.add(keras.layers.Dropout(hp.Float(
                    'dropout',
                    min_value=0.0,
                    max_value=0.4,
                    default=0.2,
                    step=0.05)))
        # Final layer produces binary prediction output
        model.add(Dense(self.nodes[-1], activation = "sigmoid"))
        # Compiles the layers of the model using a binary_crossentropy loss
        # function and stochastic gradient decent
        model.compile(loss= "binary_crossentropy", optimizer =\
                      keras.optimizers.SGD(learning_rate=hp.Choice(\
                                            'learning_rate', \
                                            values=[1e-4, 1e-5, 1e-6, 1e-7]), \
                                            momentum = \
                                            hp.Float('Momentum', .9, .99, .01)\
                                            ), metrics = ["accuracy"])
        return model
        
    def train_model(self):
        '''
        Trains the network given a set of training data containing a list of
        inputs and expected outputs
        '''
        # Stores the accuraccy after each epoch
        self.history = History()
        # Trains the model on the validation and training data
        self.model.fit(x=self.train_data, validation_data =  \
                       self.val_data, epochs = self.epochs, \
                       callbacks = [self.history, self.cp_callback],\
                       verbose=0, batch_size = self.batch_size)
        # Prints a graph of the accuracies
        self.print_accuracies()
            
    def classify_data(self, test_data):
        '''
        Takes in test data and classifies it
        '''
        # Provides predictions(output) 
        output = self.model.predict(test_data, verbose = 0)
        # Provides the predictions and the test_data accuarcy
        return output, self.model.evaluate(test_data, verbose = 0)
        
                
        
    def print_accuracies(self):
        '''
        Prints the accuracies of the model
        '''
        plt.figure(figsize = (5, 5))
        # Adds the training data accuracy to the plot
        plt.plot(self.history.epoch, \
                 self.history.history["accuracy"], label = "Training")
        # Adds the validation data accuracy to the plot
        plt.plot(self.history.epoch,\
                 self.history.history["val_accuracy"], label = "Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.xlim([0, max(self.history.epoch)])
        plt.show()
