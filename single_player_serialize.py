# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:38:20 2022

@author: Ken
"""

import game.puyo as puyo
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import functions.DQL_functions as qf
import functions.view as view
import copy
import pandas as pd
import platform
from tqdm import tqdm
import dill

eta = 0.001 #learning rate
eps = 0.30 # random action rate
gamma = 0.95 #future reward discount
batch_size = 50 #minibatch size for training
blank_memeory = True #decides whether memory lane should be filled with blanks

n_row = 4 # number of rows in the game space
n_col = 4 # number of columns in the game space
n_color = 2 #number of puyo colors
n_block = 1 #number of puto per block



#%% create game and agent
game = puyo.Puyo(n_row,n_col,n_color,n_block)
##load previous agent
infilename = ''


"""for 1 block only"""
if n_block ==1:
    output_size = n_col
    inputsize = n_color*2+(n_color+1)*n_row*n_col
else:
    raise('fix output size')

agent = keras.Sequential(
    [
        keras.Input(shape=(inputsize)),
        layers.Dense(50, activation="relu", name="layer1"),
        layers.Dense(10, activation="relu", name="layer2"),
        layers.Dense(output_size, name="output"),
    ]
)
agent.compile(loss='mean_squared_error', optimizer='adam')


#%% initialize memory
memory_size = 5000 # 100 total number of games
movemax = 100; #maximum number of moves


#populate with random move games
base_state_memory = np.zeros((memory_size, inputsize), dtype = np.int8)
action_memory  = np.zeros((memory_size,1), dtype = np.int8)
validity_memory = np.zeros((memory_size,1), dtype = np.int8)
reward_memory = np.zeros((memory_size,1))
next_state_memory = np.zeros((memory_size, inputsize), dtype = np.int8)



game.newgame()
for i in tqdm(range(memory_size)):
    
    #populate memorylane
    base_state_memory[i,:] = view.serialize(game)

    #random place
    game.place()
    game.chain()
    action_memory[i] = game.lastaction
    next_state_memory[i] = view.serialize(game)
    
    #check if game over or invalid move
    if game.valid:
        reward_memory[i] = np.log(game.score+3)
        validity_memory[i] = 1
    else:
        reward_memory[i] = -10
        validity_memory[i] = 0
        game.newgame()
    
    if game.time == movemax:
        game.newgame()




#%% training
nepoch = 1 #number of epochs
batch_size = 100
scorelist = []
totalmovelist = []

print('Start training')

"""for 1 block only"""
if n_block ==1:
    moveref = np.arange(n_col)
else:
    raise('fix moveref')

# movelog = np.zeros(nepoch,memory_size,dtype=np.int8)

for epoch in range(nepoch):
    
    for i in tqdm(range(memory_size)):
        #load into game state
        viewstate = base_state_memory[i]
        game.current_block, game.next_block,game.state = view.deserialize(viewstate, n_block, n_color, n_row, n_col)
        game.score = 0

        #with chance of eps, take random action
        if np.random.rand(1)<eps:
            game.place()
             
        else:
            #create agent_viewstate from the snapshot in memory
            #move forward in Q
            Qval = agent.predict(viewstate.reshape(1,-1),verbose=0)
            #choose the move with highest Q
            move = np.argmax(Qval)
            game.place(move=move)
            
        game.chain()   
        
        
        action_memory[i] = game.lastaction
        next_state_memory[i] = view.serialize(game)
        
        #check if game over or invalid move
        if game.valid:
            reward_memory[i] = np.log(game.score+1)
            validity_memory[i] = 1
        else:
            reward_memory[i] = -10
            validity_memory[i] = 0
            game.newgame()
            
        ###gradient descent   
        
        
        #generate batch indexes
        inds = np.random.choice(memory_size-1, batch_size)
        
        target = agent.predict(base_state_memory[inds], verbose = 0) #the qvals of base state
        actions = action_memory[inds]
        rewards = reward_memory[inds]
        validity = validity_memory[inds]
        Qval_next = agent.predict(next_state_memory[inds], verbose = 0)
        
        #update the entry in target, from the action taken and rewards observed
        #if not terminal(valid = 1), add the max qval_next with discount gamma
        target[np.arange(batch_size),actions.flatten()] = rewards.flatten()+gamma*validity.flatten()*Qval_next.max(axis=1)

      
        #feed it for gradient descent   
        loss = agent.train_on_batch(base_state_memory[inds], target)

        if (100*i/memory_size)%1==0:
            print(' ')
            print('Epoch {}, {}% though memory lane'.format(epoch,100*i/memory_size))
            print('Loss: {}'.format(loss))
         
    #at the end of each epoch, get the average score and moves before death
    bestscore=0
    totalmoves=0
    for ii in range(10):
        movecount = 0
        game.newgame()
        for iii in range(500):
            
            #calculate new action
            viewstate = view.serialize(game)
            #move forward in Q
            Qval = agent.predict(viewstate.reshape(1,-1))[0]

            #choose the move with highest Q
            move = np.argmax(Qval)

            game.place(move=move)
            game.chain()   
            
            #if valid, +1 to movecount
            if game.valid:
                movecount+=1
            else:
                break

        print(game.state)
        print(' ')
        bestscore= max(bestscore,game.totalscore)
        totalmoves = max(totalmoves,movecount)
    
    print('Epoch {} best score: {}'.format(epoch, bestscore))
    print('Epoch {} max moves: {}'.format(epoch, totalmoves))
    

    scorelist.append(bestscore)
    totalmovelist.append(totalmoves)
    if platform.system() == 'Windows':
        agent.save('agents\\agent{}'.format(epoch))
        # with open("movelog.dill", "wb") as f:
        #     dill.dump(movelog, f)
    elif platform.system() == 'Linux':       
        agent.save('../working/agents/agent{}'.format(epoch))
        # with open("movelog.dill", "wb") as f:
        #     dill.dump(movelog, f)