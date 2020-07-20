import puyo
import numpy as np
import network_relu as nt
import DQL_functions as qf
import copy
import pandas as pd
import pickle

eta = 0.01 #learning rate
eps = 0.10 # random action rate
gamma = 0.9 #future reward discount
batch_size = 50 #minibatch size for training


#create game and agent
game = puyo.Puyo()
infilename = []
if infilename !=[]:
    infile=open('agents\\{}'.format(infilename),'rb')
    agent = pickle.load(infile)
else:
    agent = nt.Network([461,50,50,22])
blank_gs=copy.deepcopy(qf.gamestate(game))

#run random choice to create memory
N = 10000 #total number of games
movemax = 500; #maximum number of moves
nepoch = 30 #number of epochs

#initialize memory lane
memory_lane = np.ndarray(N*movemax,dtype=np.object)
for i in range(len(memory_lane)):
    memory_lane[i] = qf.memory(blank_gs,0,0,blank_gs)
    
#populate with random move games
for i in range(len(memory_lane)):

    #create new random state
    height = np.random.randint(2,9)
    game.state = np.concatenate((np.zeros(13-height,6),np.random.randint(0,6,size=(height,6))))
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
        memory_lane[i].reward = -10
    memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))
    
    #if game over, reward = -10000
    if game.state[11,2]!=0:
        memory_lane[i].reward = -100
        game.newgame()
    elif game.time == movemax-1:
        game.newgame()

# training
        
scorelist = []
totalmovelist = []


print('Start training')
moveref = np.delete(np.arange(24),[1,23])
for epoch in range(nepoch):
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
             viewstate = qf.agent_view(memory_lane[i].cur_gs)
             #move forward in Q
             Qval = agent.forward(viewstate)
             #choose the move with highest Q
             Qlist=pd.DataFrame(list(zip(moveref,Qval)),columns=['move','Q'])
             Qlist.sort_values(by='Q',ascending=False,inplace=True)
             game.place(Qlist['move'].iloc[0])
             
         game.chain()
         memory_lane[i].action = game.lastaction
         memory_lane[i].next_gs = copy.deepcopy(qf.gamestate(game))
 
        #if the last move was valid, reward is score. If not, -100
         if game.valid:
             memory_lane[i].reward = qf.rewardmap(game.score)
         else:
             memory_lane[i].reward = -10    
         #if game over, reward = -10000
         if game.state[11,2]!=0:
             memory_lane[i].reward = -100
             
         #gradient descent   
         #sample random minibatch of snapshots in memory
         batch = np.random.choice(memory_lane,batch_size,replace=False)
         #calculate target for each
         target = np.zeros(batch_size)
         for ii in range(batch_size):
             if batch[ii].next_gs.state[11,2]!=0:
                 target[ii] = batch[ii].reward
             else:
                 #calculate the next y as r+gamma*maxQ of next state
                 viewstate = qf.agent_view(batch[ii].next_gs)
                 #move forward in Q
                 Qval = agent.forward(viewstate)
                 #choose the move with highest Q
                 Qlist=pd.DataFrame(list(zip(moveref,Qval)),columns=['move','Q'])
                 Qlist.sort_values(by='Q',ascending=False,inplace=True)
                 target[ii] = batch[ii].reward+gamma*Qlist['Q'].iloc[0]
         
         #feed it for gradient descent   
#         agent.train(batch,target,eta)
         loss = agent.train(batch,target,eta)
         
         if (100*i/len(memory_lane))%1==0:
             print(' ')
             print('Epoch {}, {}% though memory lane'.format(epoch,100*i/len(memory_lane)))
             print('Loss: {}'.format(loss))
         
    #at the end of each epoch, get the average score and moves before death
    bestscore=0
    totalmoves=0
    for ii in range(10):
        movecount = 0
        game.newgame()
        for iii in range(500):
            
            #calculate new action
            viewstate = qf.agent_view(qf.gamestate(game))
            #move forward in Q
            Qval = agent.forward(viewstate)
            #choose the move with highest Q
            Qlist=pd.DataFrame(list(zip(moveref,Qval)),columns=['move','Q'])
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
        bestscore= max(bestscore,game.totalscore)
        totalmoves = max(totalmoves,movecount)
    
    print('Epoch {} best score: {}'.format(epoch, bestscore))
    print('Epoch {} max moves: {}'.format(epoch, totalmoves))
    
    scorelist.append(bestscore)
    totalmovelist.append(totalmoves)
    outfile = open('agents\\agent{}'.format(1),'wb')
    pickle.dump(agent,outfile)
    outfile.close()
