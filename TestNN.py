from NeuralNet import NeuralNet
import ReadData
import os
import tensorflow as tf


"The main function which operates the neural network"
def main():

    "Supresses memory errors"
    tf.get_logger().setLevel('ERROR')

    "Sets checkpoint path for model history"
    checkpoint_path = "Dog_path/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    "Reads testing dataset from directory"
    testingDS = ReadData.get_testingData()
    "Reads training dataset from directory"
    trainingDS = ReadData.get_trainingData()
    "Reads validation dataset from directory"
    validationDS = ReadData.get_validationgData()
    '''Instantiates the NeuralNet object which will do the training and
    prediction'''
    model = NeuralNet(checkpoint_path,[180, 180,3], train_data = trainingDS,\
                      val_data = validationDS)
    "Trains the model on the training and validation data"
    model.train_model()
    '''
    Predicts the classification of the test data and reports the accuracy
    '''
    predictions, accuracy = model.classify_data(testingDS)
    "Prints the accuracy"
    #print(predictions)
    print("The accuracy on the test data was: " + accuracy[1])
    
main()
