#ReLU in hidden layer
#lin in outpuit layer


import numpy as np
import math
import DQL_functions as qf

class Network:
    def __init__(self,size):
        # size= array of numbers, corresponding to number of nodes in each layer
        self.nlayer = len(size) #number of layers
        #nodes have 1 at the end that will be used for bias
        self.nodes = [np.append(np.random.randn(layer),1) for layer in size]
        #weights has extra element to account for the 1 that is used for the bias. weight[i] is applied to nodes[i]
        self.weights = [np.random.randn(size[i+1],size[i]+1)*np.sqrt(2/(size[i]+1)) for i in range(self.nlayer-1)]
        self.z = [np.zeros(layer) for layer in size]
        self.delta = [np.zeros(layer) for layer in size]
        
    
    def forward(self,test_dat):
        #returns the values at the end
        self.nodes[0][0:-1]=test_dat
        for i in range(1,self.nlayer):
            self.z[i]=self.weights[i-1]@self.nodes[i-1]
            if i==(self.nlayer-1):
                #self.nodes[i][0:-1] = sigmoid(self.z[i])
                self.nodes[i][0:-1] = self.z[i]
            else:
                self.nodes[i][0:-1] = relu(self.z[i])
        #output layer

        return self.nodes[-1][0:-1]
    
    #run the model for some test_dat. return cost function
    #target is what it should be(label)
    def test(self,data,ans):
        test_dat = data
        ans_list = ans
        n = len(test_dat)
        counter= 0
        for i in range(n):
            image = np.divide(test_dat[i],256)
            guess=self.forward(image)
            answer = ans_list[i]
            #print(np.argmax(guess))
            if np.argmax(guess)==answer:
                counter=counter+1
        return counter,n

    def train(self,data,target,eta):
        #y is target
        #data is in class memory. contains cur_gs, action, reward, next_gs(not needed here)
        batch_size = len(data)
        dcdw=[np.zeros(l.shape) for l in self.weights]

        for ii in range(0,self.nlayer-1):
            dcdw[ii].fill(0)
        loss=0

        for ii in range(batch_size):
            viewstate = qf.agent_view(data[ii].cur_gs)
            y =  target[ii]
            guess=self.forward(viewstate)
            for iii in range(len(self.delta[-1])):
                self.delta[-1][iii] = 0
            #self.delta[-1] = (guess-y)*sigprime(self.z[-1])
            self.delta[-1][actioni(data[ii].action)] = (guess[actioni(data[ii].action)]-y)   
            loss = loss+((guess[actioni(data[ii].action)]-y)**2)/(batch_size)
            #back propagate delta's
            for iii in range(self.nlayer-2,-1,-1):
                self.delta[iii]=np.multiply(self.weights[iii][:,:-1].T@self.delta[iii+1],reluprime(self.z[iii]))
            #update weights    
            for iii in range(0,self.nlayer-1):
                dcdw[iii] = dcdw[iii]-(eta/batch_size)*(self.delta[iii+1][np.newaxis].T@self.nodes[iii][np.newaxis])
        self.weights = [self.weights[n]+dcdw[n] for n in range(self.nlayer-1)]
        return loss
        
        
def sigmoid(z):
    return 1/(1+np.exp(-z))

#def softmax(z):
#    return sigmoid(z)/sum(sigmoid(z))

def relu(z):
    return np.maximum(0,z)
    
def sigprime(z):
    return sigmoid(z)*(1-sigmoid(z))  
        
def reluprime(z):
    return z>0

def actioni(action):
    moveref = np.delete(np.arange(24),[1,23])
    return int(np.where(moveref==action)[0])