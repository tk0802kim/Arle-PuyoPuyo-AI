import numpy as np
import math
import copy


class Puyo:
    def __init__(self,state=np.zeros((13,6),dtype=np.int),current_block=np.sort(np.random.randint(1,6,size=2)),next_block=np.sort(np.random.randint(1,6,size=2))):
        
        #blocks contain 2 puyos, sorted in order
        #0 means empty, 1~5 is colored puyo(red, green, blue,yellow,purple), 6 is garbage
        #bottommoost row is index 0
        #state will be mapeed to agent_state to be sent to the agent
        #attack is the number of garbage puyo that will be dropped after the next step. negative means garbage puyos on me next turn.
        #tp is the target point(points needed per garbage puyo)
        #nl is the nuisance leftover
        #time is the number of links ran, to show how long the game has been played
        #gb is the group bonus to be used in score calculation
        self.state = state
        self.current_block = current_block
        self.next_block = next_block
        self.score = 0
        self.attack = 0
        self.tp = 70
        self.nl = 0
        self.time = 0
        self.totalscore = 0
        self.lastaction = -1
        self.valid = -1
  
    
    #places puyo, then updates current_block and next_block    
    def place(self,move=None):
        if move == None:
            move=int(np.random.choice(np.delete(np.arange(24),[1,23]),1,False))
        elif move == 1 or move == 23:
            raise Exception('Invalid move')
        
        #move = position*4+rotation
        #position is the position of the first puyo
        #rotation 0 : second puyo directly below the first
        #rotation 1~3: 90 CW rotaion of second puyo        
        self.lastaction = move
        position = move//4
        rotation = move%4
        if rotation == 0: #if vertical, check if there are two spaces in the column
            if np.array_equal(self.state[11:13,position],[0,0]):
                self.state[11:13,position] = np.flip(self.current_block,0)
                self.valid = 1
            else:
                self.valid = 0
        elif rotation == 2:
            if np.array_equal(self.state[11:13,position],[0,0]):
                self.state[11:13,position] = self.current_block
                self.valid = 1
            else:
                self.valid = 0
        elif rotation == 1:
            if np.array_equal(self.state[12,(position,position-1)],[0,0]):
                self.state[12,(position,position-1)]=self.current_block
                self.valid = 1
            else:
                self.valid = 0
        elif rotation == 3:
            if np.array_equal(self.state[12,(position,position+1)],[0,0]):
                self.state[12,(position,position+1)]=self.current_block
                self.valid = 1
            else:
                self.valid = 0
        #reset scores, attack, and current and next blocks if it was valid move
        if self.valid:
            self.reset()

        
    #drop pyyos down
    def drop(self):
        for i in range(6):
            #delete all 0's, then add in the rest of zeros
            foo = np.delete(self.state[:,i],1*(self.state[:,i]==0))
            self.state[:,i] = np.append(foo,np.zeros(13-len(foo),dtype=np.int32))
            
    
    #runs the game for one step in the chain
    #drops all puyo, calculates one step in the puyo chain, deletes those puyos
    #returns the colorlist: list containing the number of each puyo blown up
    def link(self):

        #drop the puyos down
        self.drop()

        #check for connected puyos
        #create labelstate(checked list, 0 means not checked),blowstate(blowup list, 1 means destroy), label, que, colorlist,glist
        labelstate = np.zeros((13,6),dtype=np.int)
        blowstate = np.zeros((13,6),dtype=np.int)
        label = 1
        que=[]
        colorlist=np.array([False, False, False, False, False])
        glist=[]#contains size of blown up clusters
        #loop through all points
        for i in range(13):
            for ii in range(6):
                #check that point is not labeled, and is colorpuyo
                if labelstate[i,ii] ==0 and self.state[i,ii]>0 and self.state[i,ii]<6:
                    #populate que with the starting point
                    que.append([i,ii])
                    #check off point from labelstate
                    labelstate[i,ii]=label
                    #color of the current label
                    color = np.int(self.state[i,ii])
                    #garbage puyos connected to the current label puyos
                    garbagestate = np.zeros((13,6),dtype=np.int)
                    #repeat until que is empty
                    while que!=[]:
                        #get neighbor list
                        nlist = neighbor(que[0])
                        for n1,n2 in nlist:
                            #check if neighbors of the first element in que is same color and not checked off
                            if labelstate[n1,n2] == 0 and color == self.state[n1,n2]:
                                #if so, add those points to que
                                que.append([n1,n2])
                                # and label those neighbors in labelstate
                                labelstate[n1,n2] = label
                            #if it is garbage puyo, add it to garbagestate
                            elif self.state[n1,n2]==6:
                                garbagestate[n1,n2]=1
                        #remove the element from que
                        que.pop(0)
                    #check to see if n>4
                    foostate = labelstate==label
                    n=np.count_nonzero(foostate)
                    if n>=4:
                        #add the current labels and garbage state to be blown up
                        blowstate+=(foostate)
                        blowstate+=garbagestate
                        #flip the appropriate colorlist
                        colorlist[color-1]=True
                        #add n to glist
                        glist.append(n)
#                        print('Seed at [{},{}], n={}'.format(i,ii,n))
                        
                    #update label for next iteration    
                    label+=1

        #explode the connected puyos
        self.state = np.multiply(self.state,np.logical_not(blowstate))
        return colorlist, glist
    
    
    #runs the chain sequence, iterating self.link untill it returns 0 puyos destroyed
    #returns the score of the chain
    #updates current_block and next_block
    def chain(self):
        if self.valid:
            combo = 1
            while True:
                colorlist,glist = self.link()
                if np.any(glist):
                    self.score+=calc_score(combo, colorlist,glist)
    #                print('Combo: {}'.format(combo))
    #                print(colorlist)
    #                print('Attack: {}'.format(self.score/70))
                    combo+=1
                else:
                    break
            #if all clear, add 30 to attack (https://puyonexus.com/wiki/All_clear)
            if not self.state.any():
                self.score += 30*self.tp
            #update totalscore
            self.totalscore+=self.score
            #calculate nuisance
            nnp = self.score/self.tp+self.nl
            nc = math.floor(nnp) #number of garbage puyos sent
            self.nl = nnp-nc
            self.attack += nc
        #increment time even if move wasnt valid
        self.time +=1
          
       
        
    #everything that happnes after chain blowup and next block placement
    #clears score, attack, and updates current and next blocks
    def reset(self):
        self.score=0
        self.attack=0
        self.current_block = copy.deepcopy(self.next_block)
        self.next_block=np.sort(np.random.randint(1,6,size=2))
        
    def newgame(self):
        self.state = np.zeros((13,6),dtype=np.int)
        self.current_block=np.sort(np.random.randint(1,6,size=2))
        self.next_block=np.sort(np.random.randint(1,6,size=2))
        self.score = 0
        self.attack = 0
        self.tp = 70
        self.nl = 0
        self.time = 0
        self.totalscore = 0
        self.lastaction = -1
        self.valid = -1


#function to return the list of neighbors
def neighbor(loc):
    nlist = np.array([[loc[0],loc[1]+1],[loc[0]+1,loc[1]],[loc[0]-1,loc[1]],[loc[0],loc[1]-1]])
    delete_list=[]
    if loc[0] == 0:
        delete_list.append(2)
    if loc[0] == 12:
        delete_list.append(1)
    if loc[1] == 0:
        delete_list.append(3)
    if loc[1] == 5:
        delete_list.append(0)
    
    return np.delete(nlist,delete_list,0)


#calculate score
#https://puyonexus.com/wiki/Scoring
def calc_score(combo,colorlist,glist):
    
    cblist = np.array([0,3,6,12,24]) #color bonus, for number of colors destroyed
    gblist = np.array([0,2,3,4,5,6,7]) #group bonus for number of puyos in each group, 10 for bigger than 11
    
    gb=0
    for n in glist:
        if n>=11:
            gb+=10
        else:
             gb+=gblist[n-4]   
    
    cb = cblist[np.count_nonzero(colorlist)-1]
    
    if combo>=24:
        cp = 672
    elif combo>=4:
        cp = (combo-3)*32
    else:
        cp = (combo-1)*8
    
    score = 10*sum(glist)*(gb+cb+cp)
#    print(colorlist)
#    print('gb = {}'.format(gb))
#    print('cb = {}'.format(cb))
#    print('cp = {}'.format(cp))
#    print('score = {}'.format(score))
#    print('total= {}'.format(sum(glist)))
#    print('glist = {}'.format(glist))
    
    return score