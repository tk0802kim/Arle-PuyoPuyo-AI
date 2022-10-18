import numpy as np



#sets the reward map    
def rewardmap(score):
    return np.log(score+1)



def pair_ind(nrow): #gives back indexes of the middle pieces of a row, and moving outward
    ind_list = []
    if nrow%2 == 1:
        ind_list.append(np.array([nrow//2]))
        for i in range(1,nrow//2+1):
             ind_list.append(np.array([nrow//2-i,nrow//2+i]))       
             
    elif nrow%2 == 0:
        ind_list.append(np.array([int(nrow/2-1),int(nrow/2)]))
        for i in range(1,int(nrow/2)):
             ind_list.append(np.array([ind_list[0][0]-i,ind_list[0][1]+i]))  
             
    else:
         raise Exception('Issue in DQL_functions/pair_ind' )   

    return ind_list

def best_move(moveref,Qval,mirrored):
    n = len(moveref)
    Q_i = np.argsort(Qval)[-1]
    
    if mirrored == 0:
        return Q_i
    else:
        return np.flip([*range(n)])[Q_i]
    