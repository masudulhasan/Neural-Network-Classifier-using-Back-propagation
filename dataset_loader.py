import gzip

# Third-party libraries
import numpy as np
import random
from decimal import Decimal
from sklearn import cross_validation

def vectorized_result(j):
    e = np.zeros((30, 1))
    j=int(j)
    e[j] = 1.0
    return e

def load_data():
    with open('/home/masud/neural-networks-and-deep-learning/src/testdata.txt') as a_file:
        training_inputs=list()
        training_output=list()
        I=0
        for a_line in a_file:
            s = a_line.rstrip()
            parts=s.split(',')
            # print (parts)
            # print(len(parts))
            x=list()
            y=list()
            if(parts[0]=='M'):
                x.append(0)
            if (parts[0] == 'F'):
                x.append(1)
            if (parts[0] == 'I'):
                x.append(1)
            for X in range(1,len(parts)-1):
                # temp=Decimal(parts[X])
                temp=float(parts[X])
                x.append(temp)
            # print(x)
            # print(len(x))
            y.append(int(parts[len(parts)-1]))
            # print(y)
            # training_inputs.append([np.reshape(x,(8, 1)) for X in x])
            training_inputs.append(np.array(x))
            training_output.append([np.reshape(y,(1, 1)) for Y in y])

    data_train, data_test, target_train, target_test = cross_validation.train_test_split(training_inputs,training_output,
                                                                                             test_size=0.1,
                                                                                             random_state=43)
    data_train, data_val, target_train, target_val = cross_validation.train_test_split(data_train,target_train,test_size=0.1,random_state=43)
    # print(len(data_train))
    # print(target_test)
    # print(len(target_test))
    data=[]
    I=0
    for X in target_train:
        for y in X[0]:
            # print(y)
            for z in y:
                # print(z)
                data.append(z)
        # print()
    # print(data)
    training_results = [vectorized_result(y) for y in data]
    # print(type(training_results))

    training_data = zip(data_train, training_results)

    data = []
    I = 0
    for X in target_val:
        for y in X[0]:
            # print(y)
            for z in y:
                # print(z)
                data.append(z)
                # print()
    # print(data)
    validation_results = np.array(data)
    # print(type(training_results))

    validation_data = zip(data_val,validation_results)


    I = 0
    data=[]
    for X in target_test:
        for y in X[0]:
            for z in y:
                data.append(z)
    # print data
    # test_result=np.reshape(data,(len(data), 1))
    test_result=np.array(data)
    test_data = zip(data_test,test_result)
    # print type(test_result)
    # I = 0
    # for X in test_result:
    #     print X
    #     # print '\n'
    #     if (I == 5):
    #         break
    #     I += 1

    # print type(test_data)
    return (training_data, validation_data, test_data)
load_data()


