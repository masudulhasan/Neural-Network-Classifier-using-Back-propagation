from __future__ import division
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # print a
        # print self.biases
        for b, w in zip(self.biases, self.weights):
            # print b
            # X = input("dsfdsf")
            z = self.dotProduct(w, a, b)
            a = sigmoid(z)
            # print "a"
            # print a
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):

        if test_data: n_test = len(test_data)
        n = len(training_data)
        Max=-99999
        for j in xrange(epochs):
            # random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                correct=self.evaluate(test_data)
                accuracy=correct/n_test
                print (accuracy*100)
                if(accuracy>Max):
                    Max=accuracy
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
        print "Max "+ str(Max)
        return Max

    def evaluate(self, test_data):
        # for (x,y) in test_data:
        #     print "Index "+str(np.argmax(self.feedforward(x)))

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        print test_results
        return sum(int(x == y) for (x, y) in test_results)

    def update_mini_batch(self, mini_batch, eta):
        # print mini_batch
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b=[]
        for b in range(0,len(self.biases)):
            nabla_b.append(np.zeros(len(self.biases[b])))
        # print (nabla_w)
        # for X in nabla_w:
        #     print "Len " + str(len(X))
        #     print '\n'
        # print "leaving "
        # print nabla_w.shape
        for x, y in mini_batch:
            # print "X type "+str(type(x))
            # print x
            # print type(y)
            # print y
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # print "bias"
            # print delta_nabla_b
            for nb in range(0,len(nabla_b)):
                nabla_b[nb]+=delta_nabla_b[nb]
            # print "weight"
            # print delta_nabla_w
            # nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # print "bias"
            # print nabla_b
            # print "weight "+str(len(nabla_w))
            # print nabla_w
            # print self.biases

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]


        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # print biases
        for x, y in zip(biases[0:],nabla_b[0:]):
            for X, Y in zip(x[0:],y[0:]):
                # print Y
                # print X
                # print type(X)
                for Z in range(0,len(X)):
                    X[Z]=Y
                #     print "X[Z]"
                # print X
            # # print "end"

        # print "af"
        # print biases
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, biases)]
        # print len(self.biases)
        # print self.biases

    def dotProduct(self,weight,Input,bias):
        # print weight.shape
        # print len(Input)
        product=[]
        for X in weight:
            Row=np.array(X)
            # print "row"
            # print Row
            # print "input"
            # print Input
            temp=np.dot(Row,Input)
            # print "product"
            # print temp
            product.append(temp)
        # print type(bias)
        # product += bias

        data = []
        I = 0
        for X in bias:
            for y in X:
                # print(y)
                data.append(y)

        # print data
        # print product

        data=np.array(data)
        product=np.array(product)
        product+=data
        # print len(data)
        # print len(product)
        # print product
        # x=input("dsfsf")

        return product

    def weightMatrix(self,delta,activations,index):
        Matrix = [[0 for x in range(len(activations))] for y in range(len(delta))]
        j=0
        i=0

        for Y in delta:
            j=0
            for X in activations:
                Matrix[i][j]=Y*X
                j+=1
            i+=1
        # print Matrix

        # print "weihht"
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # nabla_w[index]=Matrix
        #
        j=0
        i=0
        for Y in nabla_w[index]:
            j=0
            for X in Y:
                nabla_w[index][i][j]=Matrix[i][j]
                # print str(nabla_w[index][i][j])+ str(Matrix[i][j])
                j+=1
            i+=1

        # print nabla_w[index]
        return nabla_w[index]


    def calculateDelta(self,weight,delta,sp):
        # print "weight"
        # print weight.shape[0]
        # print weight
        # print "delta"
        # print delta.shape
        # print delta
        # print type(delta)

        j = 0
        i = 0
        tempDelta=delta
        # print "temp"
        # print tempDelta
        newDelta=[]
        for Y in weight:
            j = 0
            for X in Y:
                temp=weight[i][j]*tempDelta[j]
                j += 1
            newDelta.append(temp)
            i += 1
        delta=np.array(newDelta)
        # print "after"
        # print delta
        #
        # print "Sp"
        # print len(sp)

        for i in range(0,len(delta)):
            delta[i]=delta[i]*sp[i]

        # print "final"
        # print delta

        return delta





    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # print w
            # print activation
            # for X in
            z=self.dotProduct(w,activation,b)
            # print len(z)
            zs.append(z)
            # print zs
            activation = sigmoid(z)
            # print "asd"
            # print len(activation)
            activations.append(activation)

        # backward pass
        # print "Delta "
        costDerv=np.array(self.cost_derivative(activations[-1], y))
        # print costDerv
        # print "Sigmod"
        sigDerv=np.array(sigmoid_prime(zs[-1]))
        # print sigDerv
        delta = costDerv*sigDerv
        # print "Delta"
        # print delta
        nabla_b[-1] = delta
        # print "nabla_b"
        # print  nabla_b
        # print "activations "
        # print activations[-2]

        # x = input("dsfdsf")
        # nabla_w[-1] = np.dot(delta.transpose(), activations[-2].transpose())
        nabla_w[-1]=self.weightMatrix(delta,activations[-2],-1)
        # print "nabla_w"
        # print nabla_w[-1]


        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            delta=self.calculateDelta(self.weights[-l + 1].transpose(),delta, sp)
            nabla_b[-l] = delta
            # nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            nabla_w[-l] = self.weightMatrix(delta, activations[-l-1], -l)
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        # print output_activations
        # data = np.array(output_activations)
        data = []
        for X in y:
            for Y in X:
                # print(Y)
                data.append(Y)
        # print data
        # product = np.array(product)
        # product += data
        # print output_activations
        return (output_activations - data)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))