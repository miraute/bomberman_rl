# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:36:28 2019

@author: Michael
"""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

if True:
    reward_sum, reward_mean = genfromtxt('out.csv', delimiter=',',unpack=True)
    #plt.scatter(np.arange(len(reward_sum)), reward_sum, marker='x',s=10, color='blue', label='sum')
    plt.scatter(np.arange(len(reward_mean)), reward_mean, marker='x',s=10, color='red', label='mean')
    plt.xlabel('episodes')
    plt.ylabel('summed reward')
    plt.legend()
    plt.savefig('summed_rewards.png', dpi=300)
    plt.close()
    
    weights = genfromtxt('weights.csv', delimiter=',')
    plt.plot(np.arange(weights.shape[0]), weights, alpha=.7)
    plt.xlabel('steps')
    plt.ylabel('weights')
    plt.legend(np.arange(weights.shape[1]))
    plt.savefig('weights.png', dpi=300)
    plt.close()

if False:
    ### self :    containing the arena as self.game_state['arena'] as well as the positions of bombs and opponents
    ### arena:    A 2D numpy array describing the tiles of the game board. Its entries are 1 for crates, 
    ###           -1 for stone walls and 0 for free tiles.
    ### bomb_map: A 2D numpy array with the same shape as arena containing 5 everywhere except for the tiles
    ###             where a bomb is going to explode: there it contains the countdown value
    ### start:    tuple of (x,y): the coordinates for the starting position for the search
    ### action:   string, 'BOMB' or something else, see inteded use
    #
    ### intended use: if we want to check if it is a good idea to drop a bomb (in the sense that we can outrun it)
    ###                 - pass our current position as start and action='BOMB'
    ###               e.g. if we just dropped a bomb, we can call this function with our options for the next action
    ###                 - pass our expected next position as start and action != 'BOMB'
    #
    ### return:   bool, True if we can run or hide, else False
    def can_run_or_hide(arena, bomb_map, start, action='other'):
        mapi = arena.copy()
        print(mapi)
        ### init queue
        queue = [start+(0,)]
        while len(queue) > 0:       # as long as there are tiles to check:
            x,y,i = queue.pop()     # pick the last
            mapi[x,y]=.5+i         # mark current tile as visited  
            print(mapi)
            if action=='BOMB':      # if we want to decide if we can outrun a bomb if we would drop one at start:
                if bomb_map[start]==0:
                    return False,mapi
                if abs(x-start[0])>3 or abs(y-start[1])>3 or (x != start[0] and y != start[1]):
                    return True,mapi     # outrun in straight line or hide around a corner
            else:                   # if we want to check if we can survive the next few steps (e.g. after having dropped a bomb):
                if bomb_map[x,y] == 5:      # check if we can reach a tile where there is no due explosion 
                    return True,mapi
            for xy in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]: # check adjecent tiles
                if mapi[xy]==0 and bomb_map[xy]>i+1:     # append them to the queue if they are empty and there is no explosion by the time we reach them
                    queue.append(xy+(i+1,))
        return False,mapi                # no more tiles to check and no way out found
    
    i,start,action=67,(7,3),'BOMB'
    map = 1.0*np.load('map%i.npy'%i)
    bomb_map = np.load('bmap%i.npy'%i)
    #map[start] = 2
    
    q,map = can_run_or_hide(map, bomb_map, start, action)

    print('can run or hide?', q)
    print(map)
#%%
if False:
    i,me=6,(15,15)
    map,bmap = np.load('map%i.npy'%i).T, np.load('bmap%i.npy'%i).T
    map[me[::-1]]=2
    
#%%
if False:
    map = 1.0*np.load('map.npy')
    start = (8,1)
    crate_count = 0
    x,y = start
    queue = []
    for xy in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
        if map[xy] != -1:
            queue.append(xy+(1,))
            map[xy] += 0.1
    while len(queue)>0:
        x,y,i = queue.pop()
        if map[x,y] == 1+.1*i:
            crate_count += 1
            print(f'crate found at {(x,y)}')
        if i+1 <= 3:
            if x != start[0]:
                xy = (x+np.sign(x-start[0]),y)
            elif y != start[1]:
                xy = (x,y+np.sign(y-start[1]))
            if map[xy] != -1:
                queue.append(xy+(i+1,))
                map[xy] += 0.1*(i+1)
    print(crate_count)
    
    def get_explosion_xys(start, map, bomb_power=3):
            x, y = start
            expl = [(x,y)]
            for i in range(1, bomb_power+1):
                if map[x+i,y] == -1: break
                expl.append((x+i,y))
            for i in range(1, bomb_power+1):
                if map[x-i,y] == -1: break
                expl.append((x-i,y))
            for i in range(1, bomb_power+1):
                if map[x,y+i] == -1: break
                expl.append((x,y+i))
            for i in range(1, bomb_power+1):
                if map[x,y-i] == -1: break
                expl.append((x,y-i))
    
            return np.array(expl)
    
    if False:
        def can_run_or_hide(self, state, action):
            ... # init map, a map of the current game where all obstacles have been marked
                # with value != 0 and bomb_map, a map of the current game where all 
                # explosion countdowns are marked (5 for no explosion)
            queue = [start+(0,)]
            while len(queue) > 0:       # as long as there are tiles to check:
                x,y,i = queue.pop()     # pick the last
                map[x,y]=0.5            # mark current tile as visited  
                # if we want to decide if we can outrun a bomb if we would drop one at start:                     
                if action=='BOMB':      
                    if (abs(x-start[0])>s.bomb_power or abs(y-start[1])>s.bomb_power or 
                        (x != start[0] and y != start[1])):
                        return True     # outrun in straight line or hide around a corner
                # if we want to check if we can survive the next few steps 
                #   (e.g. after having dropped a bomb):
                else:
                    # check if we can reach a tile where there is no due explosion 
                    if bomb_map[x,y] == 5:
                        return True
                for xy in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]: # check adjecent tiles
                    # append them to the queue if they are empty and there is no 
                    # explosion by the time we reach them
                    if map[xy]==0 and bomb_map[xy]>i+1: 
                        queue.append(xy+(i+1,))
            return False                # no more tiles to check and no way out found


