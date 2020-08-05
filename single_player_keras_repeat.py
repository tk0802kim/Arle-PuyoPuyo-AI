import puyo
import numpy as np
from tensorflow import keras
from keras import layers
import DQL_functions as qf
import copy
import pandas as pd
import platform
from tqdm import tqdm

eta = 0.01 #learning rate
eps = 0.30 # random action rate
gamma = 0.95 #future reward discount
batch_size = 50 #minibatch size for training
blank_memeory = True #decides whether memory lane should be filled with blanks

#create game and agent
game = puyo.Puyo()
agent = keras.Sequential(
    [
        #keras.Input(shape=(462,)),
        layers.Dense(100, activation="relu", name="layer1"),
        layers.Dense(100, activation="relu", name="layer2"),
        layers.Dense(22, name="output"),
    ]
)
agent.compile(loss='mean_squared_error', optimizer='adam')

##load previous agent
infilename = ''
if infilename !='':
    if platform.system() == 'Windows':
        agent.load_weights('agents\\agent{}'.format(infilename))
    elif platform.system() == 'Linux':
        agent.load_weights('agents/agent{}'.format(infilename))

#run random choice to create memory
N = 100 #total number of games
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
moveref = np.delete(np.arange(24),[1,23])

for epoch in range(nepoch):
    
    #populate with random move games
    for i in tqdm(range(len(memory_lane))):
    
        #create new random state
        height = np.random.randint(7,12)
        game.state = np.concatenate((np.zeros([13-height,6]),np.random.randint(0,6,size=(height,6))))
        game.chain()
        game.reset()
        
        #populate memorylane
        memory_lane[i].cur_gs = copy.deepcopy(qf.gamestate(game))
        
        #random place
        game.place()
        game.chain()
        memory_lane[i].action = game.lastaction
        #if the last move was valid, reward is score. If not, -100
        if game.valid:
            memory_lane[i].reward = qf.rewardmap(game.score)
        else:
            memory_lane[i].reward = -10000
        memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))
        
        #if game over, reward = -10000
        if game.state[11,2]!=0:
            memory_lane[i].reward = -100
    
    
    
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
            viewstate = np.reshape(qf.agent_view(memory_lane[i].cur_gs),(1,462))
            #move forward in Q
            Qval = agent(viewstate,training=False)
            #choose the move with highest Q
            Qlist=pd.DataFrame({'move':moveref,'Q':Qval[0]})
            Qlist.sort_values(by='Q',ascending=False,inplace=True)
            game.place(Qlist['move'].iloc[0])
            
        game.chain()
        memory_lane[i].action = game.lastaction
        memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))

       #if the last move was valid, reward is score. If not, negative score
        if game.valid:
            memory_lane[i].reward = qf.rewardmap(game.score)
        else:
            memory_lane[i].reward = -3000    
        #if game over, reward = negative score
        if game.state[11,2]!=0:
            memory_lane[i].reward = -1000
            
        #gradient descent   
        #sample random minibatch of snapshots in memory
        batch = np.random.choice(memory_lane,batch_size,replace=False)
        #calculate target for each
        target = np.zeros(shape=(batch_size,22))
        viewstate_cur_batch = np.zeros(shape=(batch_size,462)) #list of currernt states in agent viewstate form
        viewstate_next_batch = np.zeros(shape=(batch_size,462))
        
        for ii in range(batch_size):
            viewstate_cur_batch[ii] = np.reshape(qf.agent_view(batch[ii].cur_gs),(1,462))
            viewstate_next_batch[ii] = np.reshape(qf.agent_view(batch[ii].next_gs),(1,462))
        target = agent.predict(viewstate_cur_batch)
        Qval_next = agent.predict(viewstate_next_batch)
        for ii in range(batch_size):
            actioni = np.where(moveref==batch[ii].action)[0][0] #index in output corresponding to action taken
            
            if batch[ii].next_gs.state[11,2]!=0:
                target[ii][actioni] = batch[ii].reward
            else:
                #choose the move with highest Q
                target[ii][actioni] = batch[ii].reward+gamma*max(Qval_next[ii])
         
        #feed it for gradient descent   
        hist = agent.fit(viewstate_cur_batch, target, batch_size=batch_size, verbose=0)
         

         
        if (100*i/len(memory_lane))%1==0:
            print(' ')
            print('Epoch {}, {}% though memory lane'.format(epoch,100*i/len(memory_lane)))
            print('Loss: {}'.format(hist.history['loss']))
         
    #at the end of each epoch, get the average score and moves before death
    bestscore=0
    totalmoves=0
    for ii in range(10):
        movecount = 0
        game.newgame()
        for iii in range(500):
            
            #calculate new action
            viewstate = np.reshape(qf.agent_view(qf.gamestate(game)),(1,462))
            #move forward in Q
            Qval = agent(viewstate).numpy()
            #choose the move with highest Q
            Qlist=pd.DataFrame({'move':moveref,'Q':Qval[0]})
            Qlist.sort_values(by='Q',ascending=False,inplace=True)
            bestmove = Qlist['move'].iloc[0]
            
            #place and run the game     
            game.place(bestmove)
            game.chain()      
            
            #if valid, +1 to movecount
            if game.valid:
                movecount+=1
            
            #check if gameover
            if game.state[11,2]!=0:
                
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
        agent.save_weights('agents\\agent{}_{}'.format(infilename,epoch))
    elif platform.system() == 'Linux':       
        agent.save_weights('agents/agent{}_{}'.format(infilename,epoch))