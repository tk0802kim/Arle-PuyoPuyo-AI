#contains normalization

import puyo
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import DQL_functions as qf
import copy
import pandas as pd
import platform
from tqdm import tqdm
import dill
import time
import datetime

eta = 0.01 #learning rate
eps = 0.30 # random action rate
gamma = 0.95 #future reward discount
batch_size = 50 #minibatch size for training
blank_memeory = True #decides whether memory lane should be filled with blanks

game_rows = 4 # number of rows in the game space
game_col = 4 # number of columns in the game space
game_n_color = 2 #number of puyo colors
game_nblock = 1 #number of puto per block
hide_top_row = False #whether to hide the very top row (game_rows+1th row) from the agent

#create game and agent
game = puyo.Puyo(game_rows-(not hide_top_row),game_col,game_n_color,game_nblock)
##load previous agent
infilename = ''


"""for 1 block only"""
if game_nblock ==1:
    output_size = game_col
    moveref = np.arange(game_col)
else:
    raise('fix output size and/or moveref')

"""specify model"""

n_conv_layer = 3

input_gamestate = keras.Input(shape=(game_n_color,game_rows,game_col))
input_blocks = keras.Input(shape=(game_n_color*2))

if tf.config.list_physical_devices('GPU'):
    conv1 =  layers.Conv2D(32, kernel_size=(3,3),activation='relu',padding='same',data_format='channels_first')(input_gamestate)
    conv1 = layers.BatchNormalization()(conv1)
elif not tf.config.list_physical_devices('GPU'):
    input_gamestate_T = tf.transpose(input_gamestate, [0, 2, 3, 1])
    conv1 =  layers.Conv2D(32, kernel_size=(3,3),activation='relu',padding='same')(input_gamestate_T)
    conv1 = layers.BatchNormalization()(conv1)
else:
    raise('issue with convolution setup')
    
for i in range(1,n_conv_layer):
    if tf.config.list_physical_devices('GPU'):
        conv1 =  layers.Conv2D(32, kernel_size=(3,3),activation='relu',padding='same',data_format='channels_first')(conv1)
        conv1 = layers.BatchNormalization()(conv1)
    elif not tf.config.list_physical_devices('GPU'):
        conv1 =  layers.Conv2D(32, kernel_size=(3,3),activation='relu',padding='same')(conv1)
        conv1 = layers.BatchNormalization()(conv1)
    else:
        raise('issue with convolution setup')


flatten = layers.Flatten()(conv1)
conc = layers.concatenate([flatten, input_blocks])
middle_layer = layers.Dense(256,activation="relu")(conc)
output_layer = layers.Dense(output_size)(middle_layer)

agent = keras.Model(inputs=[input_gamestate, input_blocks], outputs=output_layer)
agent.compile(loss='mean_squared_error', optimizer='adam')



if infilename !='':
    if platform.system() == 'Windows':
        agent = keras.models.load_model('conv_agents\\agent{}'.format(infilename))
    elif platform.system() == 'Linux':
        agent = keras.models.load_model('../working/conv_agents_32_256/agent{}'.format(infilename))

#run random choice to create memory
N = 100 # 100 total number of games
movemax = 100; #maximum number of moves
nepoch = 30 #number of epochs

#initialize memory lane
blank_gs=copy.deepcopy(qf.gamestate(game))
memory_lane = np.ndarray(N*movemax,dtype=np.object)
for i in range(len(memory_lane)):
    memory_lane[i] = qf.memory(blank_gs,0,0,blank_gs)
    
# training
        
scorelist = []
totalmovelist = []


print('Start training')


    
    
movelog = np.zeros((nepoch,len(memory_lane)),dtype=np.int)


    
for epoch in range(nepoch):
    t0 = time.time()
    
    #populate with random move games/agent games
    game.newgame()
    movecounter = 0
    
    for i in tqdm(range(len(memory_lane))):
    
        #populate memorylane
        memory_lane[i].cur_gs = copy.deepcopy(qf.gamestate(game))
    
        #place
        viewstate, viewblock = qf.agent_view_conv(game,game_n_color)
        #move forward in Q
        Qval = agent.predict([viewstate,viewblock])[0]
        #choose the move with highest Q
        move = moveref[np.argsort(Qval)[-1]]
        game.place(move=move)
        game.chain()   
        memory_lane[i].action = game.lastaction
    
        #if game over or invalid move,, reward = -10
        if game.valid:
            memory_lane[i].reward = qf.rewardmap(game.score)
            memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))
            movecounter+=1
            if movecounter==movemax:
                game.newgame()
                movecounter = 0
                
        #if the last move was valid, reward is score.
        else:
            memory_lane[i].reward = -10
            game.newgame()
    
    
    for i in range(len(memory_lane)):
        #load into game state
        game.state = memory_lane[i].cur_gs.state
        game.current_block = memory_lane[i].cur_gs.current_block
        game.next_block = memory_lane[i].cur_gs.next_block
        game.score = 0
        
        #with chance of eps, take random action
        if np.random.rand(1)<eps:
            game.place()
             
        else:
            #create agent_viewstate from the snapshot in memory
            viewstate,viewblock = qf.agent_view_conv(memory_lane[i].cur_gs,game_n_color)
            #move forward in Q
            Qval = agent.predict([viewstate,viewblock])[0]

            #choose the move with highest Q
            move = moveref[np.argsort(Qval)[-1]]
            game.place(move=move)
            
        game.chain()   
 
        memory_lane[i].action = copy.deepcopy(game.lastaction)
        movelog[epoch,i] = copy.deepcopy(game.lastaction)
        memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))

        #if game over or invalid move,, reward = -10
        if game.valid:
            memory_lane[i].reward = qf.rewardmap(game.score)
            memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))
        #if the last move was valid, reward is score.
        else:
            memory_lane[i].reward = -10
            game.newgame()

            
        #gradient descent   
        #sample random minibatch of snapshots in memory
        batch = np.random.choice(memory_lane,batch_size,replace=False)
               
        #calculate target for each
        viewstate_cur_batch = np.zeros(shape=(batch_size,game_n_color,game_rows,game_col)) #list of currernt states in agent viewstate form
        viewstate_next_batch = np.zeros(shape=(batch_size,game_n_color,game_rows,game_col))
        viewblock_cur_batch = np.zeros(shape=(batch_size,game_n_color*2)) #list of currernt states in agent viewstate form
        viewblock_next_batch = np.zeros(shape=(batch_size,game_n_color*2))
        
        for ii in range(batch_size):
            viewstate_cur_batch[ii],viewblock_cur_batch[ii] = qf.agent_view_conv(batch[ii].cur_gs,game_n_color)
            viewstate_next_batch[ii],viewblock_next_batch[ii] = qf.agent_view_conv(batch[ii].next_gs,game_n_color)
        target = agent.predict([viewstate_cur_batch,viewblock_cur_batch])
        Qval_next = agent.predict([viewstate_next_batch,viewblock_next_batch])
        for ii in range(batch_size):
            actioni = np.where(moveref==batch[ii].action)[0][0] #index in output corresponding to action taken
            
            #negative reward means game over or invalid move
            if batch[ii].reward<0:
                target[ii][actioni] = batch[ii].reward
            else:
                #choose the move with highest Q
                target[ii][actioni] = batch[ii].reward+gamma*max(Qval_next[ii])
         
        #feed it for gradient descent   
        loss = agent.train_on_batch([viewstate_cur_batch,viewblock_cur_batch], target)

        if (100*i/len(memory_lane))%1==0:
            t1 = time.time()
            print(' ')
            print(datetime.timedelta(seconds=round(t1-t0)))
            print('Epoch {}, {}% though memory lane'.format(epoch,100*i/len(memory_lane)))
            print('Loss: {}'.format(loss))
         
    #at the end of each epoch, get the average score and moves before death
    bestscore=0
    totalmoves=0
    for ii in range(10):
        movecount = 0
        game.newgame()
        for iii in range(500):
            
            #create agent_viewstate from the snapshot in memory
            viewstate,viewblock = qf.agent_view_conv(game,game_n_color)
            #move forward in Q
            Qval = agent.predict([viewstate,viewblock])[0]

            #choose the move with highest Q
            move = moveref[np.argsort(Qval)[-1]]
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
        agent.save('conv_agents\\agent{}'.format(epoch))
        with open("movelog.dill", "wb") as f:
            dill.dump(movelog, f)
    elif platform.system() == 'Linux':       
        agent.save('../working/conv_agents/agent{}'.format(epoch))
        with open("movelog.dill", "wb") as f:
            dill.dump(movelog, f)