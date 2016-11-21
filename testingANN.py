# from src.dataset_loader import load_data
# from src.mnist_loader import load_data_wrapper
# from src.network import Network
def main():
    # import  mnist_loader as mn
    # training_data, validation_data, test_data = mn.load_data_wrapper()
    # print(type(test_data))
    import dataset_loader as dl
    training_data, validation_data, test_data = dl.load_data()

    # import network
    # net = network.Network([784, 30, 10])
    # net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    Max=-9999
    alpha=3
    neuronNumber=30
    for X in range(30,100,5):
        import nn
        net = nn.Network([8,X, 30])
        # print len(net.biases[1])
        # print net.weights
        Y=0.1
        for y in range(0,100):
            accuracy=net.SGD(training_data, 20, 10, Y,validation_data)
            Y+=.1
            if(Y==3.0):
                break
            print "Accuracy "+str(accuracy)
            if(accuracy>Max):
                alpha=Y
                neuronNumber=X
    import nn
    net = nn.Network([8, neuronNumber, 30])
    # print len(net.biases[1])
    # print net.weights

    accuracy = net.SGD(training_data, 20, 10, alpha, test_data)
    print "TestAccuracy " + str(accuracy)
    # import numpy as np
    # # e = np.zeros((10, 2))
    # # print e
    # tr_d, va_d, te_d = mnist_loader.load_data()
    # training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # print training_inputs

main()