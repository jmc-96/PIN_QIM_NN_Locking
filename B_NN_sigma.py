# Neural Network klasa

import random
import numpy as np

def sigma(z):
    return 1.0/(1.0 + np.exp(-z))

def sigma_prim(z):
    # Prvi izvod sigma funkcije
    return sigma(z)*(1 - sigma(z))

# %% NNEt Klasa
class NNet(object):
    
    def __init__(self, sizes):
        self.n_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
   
    def feedforward(self, a):
        # Daje izlaz mreze ako je "a" ulaz."        
        for b, w in zip(self.biases, self.weights):
            a = sigma(np.dot(w, a) + b)
        return a
    
    def provera(self, test):
        rezultat_testa = [(np.argmax(self.feedforward(x)), y)
                          for (x, y) in test]
        return sum(int(x == y) for (x, y) in rezultat_testa)
    
    def cost_(self, izl_aktivacije, y):
        return (izl_aktivacije - y)

    def SGD(self, trening, epoha, mini_batch_size, eta, test = None):
        # "eta" je brzina ucenja (learning rate). 
        #Ako su dati "test" podaci, NN se automatski procenjuje posle svake
        #epohe i stampa pojedinacne rezultate. Usporen je proces ucenja!
        
        if test: 
            test = [(x, y) for (x,y) in test]
            n_test = len(test)
        trening = [(x, y) for (x,y) in trening]
        n = len(trening)
        for j in range(epoha):
            random.shuffle(trening)
            mini_batches = [trening[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test:
                print('Epoha {0}: {1} / {2}'.format(j, self.provera(test), n_test))
            else:
                print('Epoha {0} je zavrsena'. format(j))
              
    def update_mini_batch(self, mini_batch, eta):
        # Updateuje "weight" i "bias" vrednosti primenjujuci gradient descent
        # u backpropagation procesu.
        
        W = [np.zeros(w.shape) for w in self.weights]
        B = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            
            # Za vrednosti iz mini batcha pozivamo backpropagation funkciju
            delta_W, delta_B = self.backprop(x,y)
            
            W = [nw + dnw for nw, dnw in zip(W, delta_W)]
            B = [nb + dnb for nb, dnb in zip(B, delta_B)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, W)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, B)]

    def backprop(self, x, y):
        # Backpropagation proces
        W = [np.zeros(w.shape) for w in self.weights]
        B = [np.zeros(b.shape) for b in self.biases]
        
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z =  np.dot(w, activation) + b
            zs.append(z)
            activation = sigma(z)
            activations.append(activation)
        
        # backward pass    
        delta = self.cost_(activations[-1], y) * sigma_prim(zs[-1])
        B[-1] = delta
        W[-1] = np.dot(delta, activations[-2].transpose())
        
        for k in range(2, self.n_layers):
            z = zs[-k]
            sp = sigma_prim(z)
            delta = np.dot(self.weights[-k+1].transpose(), delta) * sp
            B[-k] = delta
            W[-k] = np.dot(delta, activations[-k -1].transpose())
        
        return (W, B)      

# %% Kraj NNet
