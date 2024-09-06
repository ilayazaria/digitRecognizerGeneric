import os.path
import pickle

from PIL import Image
from keras.datasets import mnist
import datetime
from sgd import StochasticGradientDescent


def main():
    if os.path.exists('./nn-trained.pkl'):
        print("Loading the existing neural network...")
        with open('nn-trained.pkl', 'rb') as f:
            sgd = pickle.load(f)
    else:
        print("Creating the neural network...")
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        sgd = StochasticGradientDescent(train_X, train_y)
    print(datetime.datetime.now())
    sgd.train()
    with open('nn-trained.pkl', 'wb') as f:
        pickle.dump(sgd, f)


main()
