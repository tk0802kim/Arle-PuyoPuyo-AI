# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:59:48 2022

@author: Ken
"""

import numpy as np
import functions.DQL_functions as qf


#find block index that corresponds to the given block
def block_ind(block):
    if len(block) ==1:
        return block
    else:
        raise Exception('size 2 blocks not implemented!' )   
    #this is 2block stuff
    # block_list = np.array([[1,1],[1,2],[1,3],[1,4],[1,5],[2,2],[2,3],[2,4],[2,5],[3,3],[3,4],[3,5],[4,4],[4,5],[5,5]])
    # for i in range(15):
    #         if np.array_equal(block_list[i],block):
    #             return i
    # raise Exception('block not found')


def ind_block(serial_block, n_block, n_color):
    if n_block == 1:
        for color, i in enumerate(serial_block):
            if i:
                return color+1
    else:
        raise Exception('size 2 blocks not implemented!')   


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


#game state includes: current block, next block, state
#generates a view of the game available to the agent
#[0:block_encode_length] = current block, [block_encode_length:2*block_encode_length] = next block,
#[2*block_encode_length:2* block_encode_length+viewspace_size] = empty. Next block_encode_length slots = red, etc
#total length = inputsize
#indexs from topmost row with a puyo 

def serialize(game):
    if game.n_block == 1:
        block_encode_length = game.n_color
    else:
        raise Exception('size 2 blocks not implemented!' )   
          
    #this inputsize assumes no garbage puyo's with game.n_color+1
    viewspace_size = np.multiply(*game.state.shape)
    inputsize = block_encode_length*2+(game.n_color+1)*viewspace_size #+game.n_col
    viewstate = np.zeros(inputsize,dtype=np.int8)
    
    #next 2 blocks
    cur_block_i = block_ind(game.current_block)
    next_block_i = block_ind(game.next_block)
    viewstate[cur_block_i-1] = 1
    viewstate[next_block_i-1+block_encode_length] = 1
    
    #game state
    for color in range(game.n_color+1): #also have to count empty
        start = block_encode_length*2 + viewspace_size*color
        color_vector = (game.state==color).reshape(-1).astype(np.int8)
        viewstate[start:start+viewspace_size] = color_vector
    
    return viewstate
    # # % of column full
    # for i_col in range(ncol):
    #     viewstate[0,-4+i_col] = np.mean(recolored[:,i_col]!=0) 
 


def deserialize(viewstate, n_block, n_color, n_row, n_col):
    if n_block == 1:
        block_encode_length = n_color
    else:
        raise Exception('size 2 blocks not implemented!' )   
        
    cur_block = ind_block(viewstate[0:block_encode_length], n_block, n_color)
    next_block = ind_block(viewstate[block_encode_length:2*block_encode_length], n_block, n_color)
    
    viewspace_size = n_row*n_col
    if len(viewstate)-2*block_encode_length != viewspace_size*(n_color+1):
        raise Exception('Issue deserializing: length of viewstate is inconsistent with n_block and/or n_color' )   

    viewspace = np.zeros((n_row, n_col), dtype = np.int8)
    for color in range(1, n_color+1):
        start = 2*block_encode_length+color*viewspace_size
        viewspace+=color*viewstate[start:start+viewspace_size].reshape((n_row,n_col))
    
    return np.array([cur_block]), np.array([next_block]), viewspace
        
    
def recolor(game, return_mapping = False):
   
    """
    recolor:
        recolor the puyos so that 1 is the most numerous, 2 is next.....
    priority rules:
        1. Most puyo color in viewspace
        2. color of current puyo (multiply 0.5 to count so you dont flip the previous results )
        3. color of next puyo (multiply 0.25 to count)
        4. more in first row, second row, third row....
        5. check middle in first row, next ones...second row middle, next.....
       
    This fails only if all of these are true:
        1. same number of placed puyo
        2. not favored by cur or next blocks
        3. symmetric about middle(only for even number columns)
    """
    
    viewspace = game.state
 
    rc_cur = np.zeros(gamestate.current_block.shape,dtype=int)
    rc_next = np.zeros(gamestate.next_block.shape,dtype=int)
    

    p_i=pair_ind(ncol) #gives back indexes of the middle pieces of a row, and moving outward
    
    color_sum = np.array([np.count_nonzero(viewspace==c) for c in range(1,game_n_color+1)],dtype=float) #rule 1
    
    #tie breakers
    
    if len(np.unique(color_sum)) != game_n_color:
        cur_sum = np.array([np.count_nonzero(gamestate.current_block==c) for c in range(1,game_n_color+1)],dtype=float) #rule 2
        color_sum += (1/2**tiebreaker_level)*cur_sum
        
    if len(np.unique(color_sum)) != game_n_color:
        tiebreaker_level += 1
        next_sum = np.array([np.count_nonzero(gamestate.current_block==c) for c in range(1,game_n_color+1)],dtype=float) #rule 3
        color_sum += (1/2**tiebreaker_level)*next_sum
        
    if len(np.unique(color_sum)) != game_n_color:                
        for i in range(top_i,nrow):
            tiebreaker_level += 1
            color_sum += (1/2**tiebreaker_level)*np.array([np.count_nonzero(viewspace[i,:]==c) for c in range(1,game_n_color+1)],dtype=float) #rule 4

    if len(np.unique(color_sum)) != game_n_color:                    
        for i in range(top_i,nrow):
            for inds in p_i:
                tiebreaker_level += 1
                color_sum += (1/2**tiebreaker_level)*np.array([np.count_nonzero(viewspace[i,inds]==c) for c in range(1,game_n_color+1)],dtype=float) #rule 5





    recolor_ind = np.flip(np.argsort(color_sum))+1
    color_sum_recolored = np.flip(np.sort(color_sum))
    #rc is the original color, c is the new color. So the state is going form rc->c
    for rc, c in enumerate(recolor_ind):
        recolored += (rc+1)*(viewspace==c)           #recolored gamespace
        rc_cur += (rc+1)*(gamestate.current_block==c) #recolored cur block
        rc_next += (rc+1)*(gamestate.next_block==c)   #recolored next block
    
    """
    once order is decided, check outermost in first row, higher rank should be left
        
    if there are still ties
    find lowest ranking duplicates
    with the non-duplicates, flip to right order
    then, the left most duplicates are higher ranks
    
    start here
    """
    mirrored = 0
    done = 0
    
    for ti in range(top_i,nrow): #iterate from the top most occupied state to the bottom
        for inds in reversed(p_i): #iterate from outermost
            if inds.shape[0] == 1: # if we are at the middle piece when ncol is odd, theres nothign to do
                break            
            ts = recolored[ti,:] #row being looked at now
            o_l,o_r = ts[inds[0]],ts[inds[1]]
            
            if color_sum_recolored[o_l-1] > color_sum_recolored[o_r-1]: #checking if the left side puyo's color is more numerous than the right side puyo color
                done = 1
                break
            elif color_sum_recolored[o_l-1] < color_sum_recolored[o_r-1]:
                recolored = np.flip(recolored, axis=1)
                mirrored = 1
                done = 1
                break
        if done==1:
            break
    
    """this is coded for only 1 block stuff"""
    # n_orientation = ncol
    n_blocks = game_n_color #number of different blocks that can be given
    
    
    """following can be universal"""
    """
    serialize the game state
    It will be in the order of:
        color of current piece
        color of next piece
        color n serialized
        % height of each column
    """
    # +ncol at the end to signify height of each block
    viewstate = np.zeros(shape=(1,n_blocks*2+(game_n_color+1)*nrow*ncol+ncol),dtype=np.int8) #this assumes no garbage puyo's with game_n_color+1
    
    #current and next block states
    cur_block_i = block_ind(rc_cur)
    next_block_i = block_ind(rc_next)
    viewstate[0,cur_block_i-1] = 1
    viewstate[0,next_block_i-1+n_blocks] = 1
    
    # % of column full
    for i_col in range(ncol):
        viewstate[0,-4+i_col] = np.mean(recolored[:,i_col]!=0) 
    
    #game state
    for ii in range(game_n_color+1): #also have to count empty
        #coutner is counting nth row from top, i is the absolute row
        for counter, i in enumerate(range(top_i, nrow)):

            topstate = recolored[i,:]
            
            # if not any(topstate):
            #     raise('issue with viewstate')
                  
            # for slot in topstate:
            start = n_blocks*2+ii*nrow*ncol+ncol*counter
            end = n_blocks*2+ii*nrow*ncol+ncol*(counter+1)
            try:
                if mirrored:
                    viewstate[0,start:end] = ((topstate==ii)*1)[::-1]
                else:
                    viewstate[0,start:end] = (topstate==ii)*1
            except:
                print('game state serializing not working')   
                         
    return viewstate,mirrored


def agent_view(gamestate, game_n_color, hide_top_row=False):
    
    if gamestate.current_block.size>1:
        raise Exception('Haven\'t coded for nblock != 1 yet!' )
    
    nrow = gamestate.state.shape[0]-hide_top_row #dont consider last row if hidden
    ncol = gamestate.state.shape[1]
    
    #get flipstate(viewstate flipped(so the top comes first in index), topstate(first nonzero row) and top_i(index of the top row))
    top_i = 0
    for i in range(nrow): 
            if np.any(gamestate.state[i,:]):
                # flipstate = np.flip(gamestate.state[0:i+1,:],0).flatten()
                # topstate = gamestate.state[i,:]
                top_i = i
                break
    
    #%%
    """
    recolor
        1 is the most numerous, 2 is next.....
    rules:
        1. Most puyo color in viewspace
        2. color of current puyo (multiply 0.5 to count so you dont flip the previous results )
        3. color of next puyo (multiply 0.25 to count)
        4. more in first row, second row, third row....
        5. check middle in first row, next ones...second row middle, next.....
       
    This fails only if:
        1. same number of placed puyo
        2. not favored by cur or next blocks
        3. symmetric about middle
    """
    viewspace = gamestate.state[0:nrow,:]
    recolored = np.zeros((nrow,ncol),dtype=int)
    rc_cur = np.zeros(gamestate.current_block.shape,dtype=int)
    rc_next = np.zeros(gamestate.next_block.shape,dtype=int)
    
    tiebreaker_level = 1
    p_i=pair_ind(ncol) #gives back indexes of the middle pieces of a row, and moving outward
    
    color_sum = np.array([np.count_nonzero(viewspace==c) for c in range(1,game_n_color+1)],dtype=float) #rule 1
    if len(np.unique(color_sum)) != game_n_color:
        cur_sum = np.array([np.count_nonzero(gamestate.current_block==c) for c in range(1,game_n_color+1)],dtype=float) #rule 2
        color_sum += (1/2**tiebreaker_level)*cur_sum
        
        if len(np.unique(color_sum)) != game_n_color:
            tiebreaker_level += 1
            next_sum = np.array([np.count_nonzero(gamestate.current_block==c) for c in range(1,game_n_color+1)],dtype=float) #rule 3
            color_sum += (1/2**tiebreaker_level)*next_sum
            
            if len(np.unique(color_sum)) != game_n_color:                
                for i in range(top_i,nrow):
                    tiebreaker_level += 1
                    color_sum += (1/2**tiebreaker_level)*np.array([np.count_nonzero(viewspace[i,:]==c) for c in range(1,game_n_color+1)],dtype=float) #rule 4

                if len(np.unique(color_sum)) != game_n_color:                    
                    for i in range(top_i,nrow):
                        for inds in p_i:
                            tiebreaker_level += 1
                            color_sum += (1/2**tiebreaker_level)*np.array([np.count_nonzero(viewspace[i,inds]==c) for c in range(1,game_n_color+1)],dtype=float) #rule 5
    
    recolor_ind = np.flip(np.argsort(color_sum))+1
    color_sum_recolored = np.flip(np.sort(color_sum))
    #rc is the original color, c is the new color. So the state is going form rc->c
    for rc, c in enumerate(recolor_ind):
        recolored += (rc+1)*(viewspace==c)           #recolored gamespace
        rc_cur += (rc+1)*(gamestate.current_block==c) #recolored cur block
        rc_next += (rc+1)*(gamestate.next_block==c)   #recolored next block
    
    """
    once order is decided, check outermost in first row, higher rank should be left
        
    if there are still ties
    find lowest ranking duplicates
    with the non-duplicates, flip to right order
    then, the left most duplicates are higher ranks
    
    start here
    """
    mirrored = 0
    done = 0
    
    for ti in range(top_i,nrow): #iterate from the top most occupied state to the bottom
        for inds in reversed(p_i): #iterate from outermost
            if inds.shape[0] == 1: # if we are at the middle piece when ncol is odd, theres nothign to do
                break            
            ts = recolored[ti,:] #row being looked at now
            o_l,o_r = ts[inds[0]],ts[inds[1]]
            
            if color_sum_recolored[o_l-1] > color_sum_recolored[o_r-1]: #checking if the left side puyo's color is more numerous than the right side puyo color
                done = 1
                break
            elif color_sum_recolored[o_l-1] < color_sum_recolored[o_r-1]:
                recolored = np.flip(recolored, axis=1)
                mirrored = 1
                done = 1
                break
        if done==1:
            break
    
    """this is coded for only 1 block stuff"""
    # n_orientation = ncol
    n_blocks = game_n_color #number of different blocks that can be given
    
    
    """following can be universal"""
    """
    serialize the game state
    It will be in the order of:
        color of current piece
        color of next piece
        color n serialized
        % height of each column
    """
    # +ncol at the end to signify height of each block
    viewstate = np.zeros(shape=(1,n_blocks*2+(game_n_color+1)*nrow*ncol+ncol),dtype=np.int8) #this assumes no garbage puyo's with game_n_color+1
    
    #current and next block states
    cur_block_i = block_ind(rc_cur)
    next_block_i = block_ind(rc_next)
    viewstate[0,cur_block_i-1] = 1
    viewstate[0,next_block_i-1+n_blocks] = 1
    
    # % of column full
    for i_col in range(ncol):
        viewstate[0,-4+i_col] = np.mean(recolored[:,i_col]!=0) 
    
    #game state
    for ii in range(game_n_color+1): #also have to count empty
        #coutner is counting nth row from top, i is the absolute row
        for counter, i in enumerate(range(top_i, nrow)):

            topstate = recolored[i,:]
            
            # if not any(topstate):
            #     raise('issue with viewstate')
                  
            # for slot in topstate:
            start = n_blocks*2+ii*nrow*ncol+ncol*counter
            end = n_blocks*2+ii*nrow*ncol+ncol*(counter+1)
            try:
                if mirrored:
                    viewstate[0,start:end] = ((topstate==ii)*1)[::-1]
                else:
                    viewstate[0,start:end] = (topstate==ii)*1
            except:
                print('game state serializing not working')   
                         
    return viewstate,mirrored
  
#%% convolutional view
def agent_view_conv(gamestate, game_n_color, hide_top_row=False):
        
    if gamestate.current_block.size>1:
        raise Exception('Haven\'t coded for nblock != 1 yet!' )
    
    nrow = gamestate.state.shape[0]-hide_top_row #dont consider last row if hidden
    ncol = gamestate.state.shape[1]
    
    #get flipstate(viewstate flipped(so the top comes first in index), topstate(first nonzero row) and top_i(index of the top row))
    top_i = 0
    for i in reversed(range(nrow)): 
            if np.any(gamestate.state[i,:]):
                # flipstate = np.flip(gamestate.state[0:i+1,:],0).flatten()
                # topstate = gamestate.state[i,:]
                top_i = i
                break
    """
    recolor
        1 is the most numerous, 2 is next.....
    rules:
        1. Most puyo color in viewspace
        2. color of current puyo (multiply 0.5 to count so you dont flip the previous results )
        3. color of next puyo (multiply 0.25 to count)
        4. more in first row, second row, third row....
        5. check middle in first row, next ones...second row middle, next.....
       
    This fails only if:
        1. same number of placed puyo
        2. not favored by cur or next blocks
        3. symmetric about middle
    """
    viewspace = gamestate.state[0:nrow,:]
    recolored = np.zeros((nrow,ncol),dtype=int)
    rc_cur = np.zeros(gamestate.current_block.shape,dtype=int)
    rc_next = np.zeros(gamestate.next_block.shape,dtype=int)
    
    tiebreaker_level = 1
    p_i=pair_ind(ncol) #gives back indexes of the middle pieces of a row, and moving outward
    
    color_sum = np.array([np.count_nonzero(viewspace==c) for c in range(1,game_n_color+1)],dtype=float) #rule 1
    if len(np.unique(color_sum)) != game_n_color:
        cur_sum = np.array([np.count_nonzero(gamestate.current_block==c) for c in range(1,game_n_color+1)],dtype=float) #rule 2
        color_sum += (1/2**tiebreaker_level)*cur_sum
        
        if len(np.unique(color_sum)) != game_n_color:
            tiebreaker_level += 1
            next_sum = np.array([np.count_nonzero(gamestate.current_block==c) for c in range(1,game_n_color+1)],dtype=float) #rule 3
            color_sum += (1/2**tiebreaker_level)*next_sum
            
            if len(np.unique(color_sum)) != game_n_color:                
                for i in range(top_i,-1,-1):
                    tiebreaker_level += 1
                    color_sum += (1/2**tiebreaker_level)*np.array([np.count_nonzero(viewspace[i,:]==c) for c in range(1,game_n_color+1)],dtype=float) #rule 4

                if len(np.unique(color_sum)) != game_n_color:                    
                    for i in range(top_i,-1,-1):
                        for inds in p_i:
                            tiebreaker_level += 1
                            color_sum += (1/2**tiebreaker_level)*np.array([np.count_nonzero(viewspace[i,inds]==c) for c in range(1,game_n_color+1)],dtype=float) #rule 5
    
    recolor_ind = np.flip(np.argsort(color_sum))+1
    #rc is the original color, c is the new color. So the state is going form rc->c
    for rc, c in enumerate(recolor_ind):
        recolored += (rc+1)*(viewspace==c)           #recolored gamespace
        rc_cur += (rc+1)*(gamestate.current_block==c) #recolored cur block
        rc_next += (rc+1)*(gamestate.next_block==c)   #recolored next block
        
    viewstate = np.zeros(shape=(1,game_n_color,nrow,ncol),dtype=int)
    for i in range(game_n_color):
        viewstate[0,i] = (recolored==(i+1))
        
    # viewstate = np.zeros(shape=(game_n_color,nrow,ncol),dtype=int)
    # for i in range(game_n_color):
    #     viewstate.append(recolored==(i+1))
    
    """this is coded for only 1 block stuff"""
    # n_orientation = ncol
    n_blocks = game_n_color #number of different blocks that can be given
    viewblocks = np.zeros(shape=(1,n_blocks*2),dtype=int)
    viewblocks[0,rc_cur-1] = 1
    viewblocks[0,n_blocks+rc_next-1] = 1
    
    return viewstate, viewblocks
