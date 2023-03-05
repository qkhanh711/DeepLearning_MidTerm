import os
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache


def relu_backward(dA, cache):  
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    Z = cache    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)  
    return dZ   


def initialize_parameters_deep(layer_dims):   
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation, keep_prob, drop_out=True):

    if activation == "sigmoid":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        D = None
    
    elif activation == "relu":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        if drop_out:
            D = np.random.rand(A.shape[0], A.shape[1])    
            D = D < keep_prob                            
            A = A * D                                   
            A = A / keep_prob 
        else:
            D = None
        
    cache = (D, linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters, keep_probs, drop_out=True):

    caches = []
    A = X
    L = len(parameters) // 2                  
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation = "relu",
                                             keep_prob=keep_probs[l-1],
                                             drop_out=drop_out
                                             )
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], 
                                             parameters['b' + str(L)], 
                                             activation = "sigmoid",
                                             keep_prob=None,
                                             drop_out=False
                                             )
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL, Y, parameters, lambd):
    
    m = Y.shape[1]
    
    L = len(parameters) // 2
    Ws = [parameters['W' + str(i)] for i in range(1, L+1)]
    
    L2_regularization_cost = lambd * (np.sum([np.sum(np.square(W)) for W in Ws])) / (2 * m)
    cross_entropy_cost = np.squeeze((1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)))
    cost = cross_entropy_cost + L2_regularization_cost    
    
    return cost

def linear_backward(dZ, cache, lambd):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T) + (lambd * W) / m
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, lambd, activation):

    D, linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, lambd, drop_out=True):

    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, 
                                                                                                    current_cache, 
                                                                                                    lambd,
                                                                                                    activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        dA = grads["dA" + str(l + 1)]
        current_cache = caches[l]
        if drop_out:
            D = current_cache[0]
            dA = dA*D
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA, 
                                                                    current_cache, 
                                                                    lambd,
                                                                    activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):
    keep_probs = [1,1,1,1]
    m = X.shape[1]
    n = len(parameters) // 2 
    
    probas, caches = L_model_forward(X, parameters, keep_probs= keep_probs, drop_out= False)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            probas[0,i] = 1
        else:
            probas[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((probas == y)/m)))
        
    return probas

def print_mislabeled_images(classes, X, y, p):

    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) 
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))


def load_data(root, set_type):
    images = []
    labels = []

    path = glob.glob(root + "\\" + set_type + "/*")
    
    for cl in path:
        cl = os.path.basename(cl)

        label = 1 if cl=="Car" else 0
        path_2 = glob.glob(root + "\\" + set_type + "\\" + cl + "\\*")
        for img in path_2:
            labels.append(label)
            img = cv2.imread(img)
            img = cv2.resize(img, (64, 64))
            images.append(img)
    images = np.array(images)
    labels = np.array([labels]).T
    
    permutation = list(np.random.permutation(labels.shape[0]))
    labels = labels[permutation, :]
    images = images[permutation, :]
    
    return images, labels.T