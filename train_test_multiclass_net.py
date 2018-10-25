from sklearn.datasets import load_iris
import numpy as np


def train_test():

    # load the data
    iris = load_iris()
    X = iris.data[:, 0:4]  # X for the set of features
    y = iris.target  # for the set of labels
    print(np.unique(y))  # to print the labels

    # One Hot Encode Y: necessary for classification. It seems that this step encode the labels (1, 2, 3)
    # into binary coding
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(y)

    # train the network
    neural_network = create_network()
    neural_network.fit(X, Y, epochs=500, batch_size=10)

    # test the model
    np.set_printoptions(suppress=True)
    predictions = neural_network.predict(X[0:10], batch_size=32, verbose=0)
    print(predictions)


def create_network():
    """
    This function defines the structure of the neural network
    :return:
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD

    model = Sequential()
    model.add(Dense(5, input_shape=(4, ), activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # stochastic gradient descent
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def main():
    train_test()


if __name__ == '__main__':
    main()
