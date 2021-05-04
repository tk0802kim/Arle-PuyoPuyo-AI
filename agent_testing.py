import puyo
import numpy as np
import DQL_functions as qf
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 1000
movelimit=500

game_rows = 4 # number of rows in the game space
game_col = 4 # number of columns in the game space
game_n_color = 2 #number of puyo colors
game_nblock = 1 #number of puto per block
hide_top_row = False #whether to hide the very top row (game_rows+1th row) from the agent
moveref = np.arange(game_col)

#load agents
agents = {}
for i in [0,*range(9,30,10)]:
    agents['p'+str(i)] = keras.models.load_model('../working/conv_agents_32_256_predicted_move_training/agent{}'.format(i))
    agents['r'+str(i)] = keras.models.load_model('../working/conv_agents_32_256_random_move_training/agent{}'.format(i))

#create games
games = {}
for i in range(batch_size):
    games[i] = puyo.Puyo(game_rows-(not hide_top_row),game_col,game_n_color,game_nblock)
    
#run sym  
scoredict={}
movedict={}

for key in agents.keys():
    
    agent = agents[key]

    #reset scores and game
    for i in range(batch_size): games[i].newgame()
    scorelist=np.zeros(batch_size)
    movecount=np.zeros(batch_size)

    for m in tqdm(range(movelimit)):
        viewstate_batch = np.zeros(shape=(batch_size,game_n_color,game_rows,game_col)) #list of currernt states in agent viewstate form
        viewblock_batch = np.zeros(shape=(batch_size,game_n_color*2)) #list of currernt states in agent viewstate form

        for i in range(batch_size):
            viewstate_batch[i],viewblock_batch[i] = qf.agent_view_conv(games[i],game_n_color)

        Qvals = agent.predict((viewstate_batch,viewblock_batch))
        moves=  [moveref[np.argsort(Q)[-1]] for Q in Qvals]

        for i in range(batch_size):
            games[i].place(moves[i])
            games[i].chain()
            if not(games[i].valid) and movecount[i]==0:
                movecount[i] = m
                scorelist[i] = games[i].totalscore
    
    for i in range(batch_size):
        if movecount[i]==0:
            movecount[i] = m
            scorelist[i] = games[i].totalscore
            
    scoredict[key] = scorelist
    movedict[key] = movecount

    
# random
for i in range(batch_size): games[i].newgame()
scorelist=np.zeros(batch_size)
movecount=np.zeros(batch_size)
for m in range(movelimit):
    for i in range(batch_size):
        games[i].place()
        games[i].chain()
        if not(games[i].valid) and movecount[i]==0:
            movecount[i] = m
            scorelist[i] = games[i].totalscore

for i in range(batch_size):
    if movecount[i]==0:
        movecount[i] = m
        scorelist[i] = games[i].totalscore

scoredict['random'] = scorelist
movedict['random'] = movecount

    

"""graph"""    

keys = ['random','r0','r9','r19','r29','p0','p9','p19','p29']

figure,axes = plt.subplots(3,figsize=(5,10))

#score
axes[0].boxplot([scoredict[key] for key in keys],labels = keys)
axes[0].set_title('Score')
#moves
axes[1].boxplot([movedict[key] for key in keys],labels = keys)
axes[1].set_title('Moves before game over')
#maxed out games
axes[2].bar(range(len(keys)),[sum(movedict[key]==(movelimit-1))/movelimit for key in keys])
axes[2].set_title('% of games that reached {} moves'.format(movelimit))
axes[2].set_xticks = keys
plt.savefig('agent_summary_r_vs_p.pdf')
plt.show()


