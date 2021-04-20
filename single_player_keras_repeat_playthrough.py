import puyo
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
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

game_rows = 4 # number of rows in the game space
game_col = 4 # number of columns in the game space
game_n_color = 2 #number of puyo colors
game_nblock = 1 #number of puto per block
hide_top_row = False #whether to hide the very top row (game_rows+1th row) from the agent

#create game and agent
game = puyo.Puyo(game_rows-(not hide_top_row),game_col,game_n_color,game_nblock)
##load previous agent
infilename = ''

input_size = game_n_color*2+(game_n_color+1)*(game_rows-hide_top_row)*game_col

"""for 1 block only"""
if game_nblock ==1:
    output_size = game_col
else:
    raise('fix output size')

agent = keras.Sequential(
    [
        keras.Input(shape=(input_size)),
        layers.Dense(input_size//1, activation="relu", name="layer1"),
        layers.Dense(input_size//2, activation="relu", name="layer2"),
        layers.Dense(output_size, name="output"),
    ]
)
agent.compile(loss='mean_squared_error', optimizer='adam')

if infilename !='':
    if platform.system() == 'Windows':
        agent = keras.models.load_model('agents\\agent{}'.format(infilename))
    elif platform.system() == 'Linux':
        agent = keras.models.load_model('../working/agents/agent{}'.format(infilename))

#run random choice to create memory
N = 10 # 100 total number of games
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

"""for 1 block only"""
if game_nblock ==1:
    moveref = np.arange(game_col)
else:
    raise('fix moveref')
    
    
movelog = np.zeros((nepoch,len(memory_lane)),dtype=np.int)

for epoch in range(nepoch):
    
    #populate with random move games
    game.newgame()
    for i in tqdm(range(len(memory_lane))):
        
        #populate memorylane
        memory_lane[i].cur_gs = copy.deepcopy(qf.gamestate(game))
        
        #random place
        game.place()
        game.chain()
        memory_lane[i].action = game.lastaction
        
        #if game over, reward = -10000
        if game.state[-1,game_col//2]!=0:
            memory_lane[i].reward = -10
            game.newgame()
        
        
        #if the last move was valid, reward is score. If not, -100
        elif game.valid:
            memory_lane[i].reward = qf.rewardmap(game.score)
            memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))
        else:
            memory_lane[i].reward = -10
            # game.state[11,2] = 1;
            memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))
            # game.newgame()


    
    

    for i in range(len(memory_lane)):
        #load into game state
        game.state = memory_lane[i].cur_gs.state
        game.current_block = memory_lane[i].cur_gs.current_block
        game.next_block = memory_lane[i].cur_gs.next_block
        game.score = 0
        
        #with chance of eps, take random action
        if np.random.rand(1)<eps:
            game.place()
            game.chain()             
        else:
            #create agent_viewstate from the snapshot in memory
            viewstate,mirrored = qf.agent_view(memory_lane[i].cur_gs,game_n_color)
            #move forward in Q
            Qval = agent.predict(viewstate)[0]

            #choose the move with highest Q
            move = moveref[np.argsort(Qval)[-1]]
            if mirrored:
                move = game_col-1-move
            game.place(move=move)
            game.chain()
            
 
        memory_lane[i].action = copy.deepcopy(game.lastaction)
        movelog[epoch,i] = copy.deepcopy(game.lastaction)
        memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))

        #if the last move was valid, reward is score. If not, negative score
        if game.state[-1,game_col//2]!=0:             #game over
            memory_lane[i].reward = -10
            game.newgame()      
        elif game.valid:
            memory_lane[i].reward = qf.rewardmap(game.score)
        else:
            memory_lane[i].reward = -10
            # game.state[11,2] = 1;

            
        #gradient descent   
        #sample random minibatch of snapshots in memory
        batch = np.random.choice(memory_lane,batch_size,replace=False)
        #calculate target for each

        viewstate_cur_batch = np.zeros(shape=(batch_size,input_size)) #list of currernt states in agent viewstate form
        viewstate_next_batch = np.zeros(shape=(batch_size,input_size))
        
        # mirrored_cur_batch == np.zeros(batch_size)
        # mirrored_next_batch == np.zeros(batch_size)

        for ii in range(batch_size):
            viewstate_cur_batch[ii],_ = qf.agent_view(batch[ii].cur_gs,game_n_color)
            viewstate_next_batch[ii],_ = qf.agent_view(batch[ii].next_gs,game_n_color)
        target = agent.predict(viewstate_cur_batch)
        Qval_next = agent.predict(viewstate_next_batch)
        for ii in range(batch_size):
            actioni = np.where(moveref==batch[ii].action)[0][0] #index in output corresponding to action taken
            
            if batch[ii].next_gs.state[-1,game_col//2]!=0:
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
            viewstate, mirrored = qf.agent_view(memory_lane[i].cur_gs,game_n_color)
            #move forward in Q
            Qval = agent.predict(viewstate)[0]

            #choose the move with highest Q
            move = moveref[np.argsort(Qval)[-1]]
            if mirrored:
                move = game_col-1-move
            game.place(move=move)
            game.chain()   
            
            #if valid, +1 to movecount
            if game.valid:
                movecount+=1
            
            #check if gameover
            if game.state[-1,game_col//2]!=0:
                
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
    elif platform.system() == 'Linux':       
        agent.save('../working/agents/agent{}'.format(epoch))