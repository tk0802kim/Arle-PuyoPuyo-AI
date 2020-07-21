import numpy as np

#game state includes: state, current block, next block.
class gamestate:
    def __init__(self,game):
        self.state = game.state
        self.current_block = game.current_block
        self.next_block = game.next_block

#generates a view of the game available to the agent
#[0,14] = current block, [15,29] = next block,
#[30:101] = empty slot. [102:173] = red slot, etc
#total length = 461(29+72*6)
#indexs from topmost row with a puyo 
def agent_view(gamestate):
    viewstate = np.zeros(461,dtype=np.int)
    cur_block_i = block_ind(gamestate.current_block)
    next_block_i = block_ind(gamestate.next_block)
    viewstate[cur_block_i] = 1
    viewstate[next_block_i+15] = 1
    for i in range(11,-1,-1): #omit 13th row because thats invisible
            if np.any(gamestate.state[i]):
                topstate = np.flip(gamestate.state[0:i+1,:],0).flatten()
                toplen=len(topstate);
                for ii in range(6):
                    for slot in topstate:
                        viewstate[30*i*72:29*i*72+toplen] = (slot==ii)*1
                return viewstate      
    return viewstate
    
#find block index that correstponds to the given block
def block_ind(block):
    block_list = np.array([[1,1],[1,2],[1,3],[1,4],[1,5],[2,2],[2,3],[2,4],[2,5],[3,3],[3,4],[3,5],[4,4],[4,5],[5,5]])
    for i in range(15):
            if np.array_equal(block_list[i],block):
                return i
    raise Exception('block not found')

#memory includes starting game state, action and reward of that action, and the resulting gamestate
class memory:
    def __init__(self, cur_gs, action, reward, next_gs):
        self.cur_gs = cur_gs
        self.action = action
        self.reward = reward
        self.next_gs = next_gs
        
#sets the reward map    
def rewardmap(score):
    return score