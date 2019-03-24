
import numpy as np
#from random import shuffle
#from time import time, sleep
from collections import deque

from settings import s
import csv



def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    
    #import os    # find out current path and working directory
    #self.logger.debug(f'path: {os.path.dirname(os.path.realpath(__file__))}, WD: {os.getcwd()}')
    
    np.random.seed()
    
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 5)
    
    self.update = 5                             # 'move': SARSA update every move or int n: n-step SARSA update after every episode
    self.state_hist = deque([],s.max_steps)     # state history, needed for SARSA learning
    self.action_hist = deque([],s.max_steps)    # action history, -------"------------
    self.reward_sum = 0.                        # sum of rewards per episode, for monitoring
    self.reward_hist = deque([],s.max_steps)    # list of all rewards during current episode
    self.out_file_weight = 'weights.csv'    # csv-file where weights are appended to after update (step)
    self.out_file_reward = 'out.csv'        # csv-file where reward_sum is appended to after each episode
    
    self.file_weights = 'weights.npy'
    self.weights = np.load(self.file_weights)      # load trained weights
    self.weights_len = 9
    self.gamma = 0.9            # discount factor
    self.alpha = 0.001          # learning rate for gradient descent SARSA
    self.eps_start = 0.0        # for epsilon-greedy action selection   # 0.9
    self.eps_end = 0.00         # and annealing of epsilon              # 0.05
    self.eps_decay = 20
    self.epsilon = self.eps_start
    
    self.last_map = deque([],1)
    self.last_bmap= deque([],1)
    self.me = ()
    
    try:                        # initialize episode counter:
        self.episode_nr
    except AttributeError:
        self.episode_nr = 0

def get_explosion_xys(start, map, bomb_power=3):
    """
    returns all tiles hit by an explosion starting at start given a 2d map of the game 
       where walls are indicated by -1        
    """
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
    

def can_run_or_hide(self, state, start, action='other'):
    """
     self :    containing the arena as self.game_state['arena'] as well as the positions of bombs and opponents
     arena:    A 2D numpy array describing the tiles of the game board. Its entries are 1 for crates, 
               -1 for stone walls and 0 for free tiles.
     bomb_map: A 2D numpy array with the same shape as arena containing 5 everywhere except for the tiles
                 where a bomb is going to explode: there it contains the countdown value
     start:    tuple of (x,y): the coordinates for the starting position for the search
     action:   string, 'BOMB' or something else, see inteded use
    
     intended use: if we want to check if it is a good idea to drop a bomb (in the sense that we can outrun it)
                     - pass our current position as start and action='BOMB'
                   e.g. if we just dropped a bomb, we can call this function with our options for the next action
                     - pass our expected next position as start and action != 'BOMB'
    
     return:   bool, True if we can run or hide, else False
    """
    game_state,_,bomb_map = state
    if action=='BOMB' and bomb_map[start]==0: # first, check if staying at the same place is an option
        return False
    ### prepare a map of the game
    map = game_state['arena'].copy()
    obstacles = [(x,y) for (x,y,t) in game_state['bombs']]
    obstacles.extend([(x,y) for (x,y,n,b,s) in game_state['others']])
    for o in obstacles:
        map[o] = -2
    self.last_map.append(map)
    self.last_bmap.append(bomb_map)
    
    ### init queue
    queue = [start+(0,)]
    while len(queue) > 0:       # as long as there are tiles to check:
        #self.logger.debug(f'queue={queue}')
        x,y,i = queue.pop()     # pick the last
        map[x,y]=0.5#+i         # mark current tile as visited  
        #self.logger.debug(f'(x,y,i)={(x,y,i)}')
        if action=='BOMB':      # if we want to decide if we can outrun a bomb if we would drop one at start:
            if abs(x-start[0])>s.bomb_power or abs(y-start[1])>s.bomb_power or (x != start[0] and y != start[1]):
                return True     # outrun in straight line or hide around a corner
        else:                   # if we want to check if we can survive the next few steps (e.g. after having dropped a bomb):
            if bomb_map[x,y] == 5:      # check if we can reach a tile where there is no due explosion 
                return True
        for xy in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]: # check adjecent tiles
            #self.logger.debug(f'xy={xy}')
            #self.logger.debug(f'map[xy]={map[xy]}, bomb_map[xy]={bomb_map[xy]}')
            try:
                if map[xy]==0 and bomb_map[xy]>i+1:     # append them to the queue if they are empty and there is no explosion by the time we reach them
                    queue.append(xy+(i+1,))
            except Exception as e:
                self.logger.debug(f'ERROR in can_run_or_hide(): start={start}, xy={xy}, me={self.me}\n{self.last_map}\n{self.last_bmap}\n Orig. message:\n {e}')
                pass
    return False                # no more tiles to check and no way out found


def distance(pos1, pos2):
    return np.sum(np.square(np.subtract(pos1,pos2)), axis=-1)


def funcs(self, state, action):
    """
     contains the coefficient functions that are multplying the weights to (linearly) approximate Q(s,a)
     input:  (object) self: needed for logging functionality, constants, ...
             (list) state: containing all infos about current state of the game, (string) action
     return: (np.array) coefficient array (length: self.weights.shape) 
    """
    game_state, coordinate_history, bomb_map = state
    arena = game_state['arena']
    x, y, _, bombs_left, _ = game_state['self']

    others = [(x,y) for (x,y,n,b,s) in game_state['others']]
    dict_actions = {'LEFT': (x-1,y), 'RIGHT': (x+1,y), 'UP': (x,y-1), 'DOWN': (x,y+1), 'WAIT': (x,y), 'BOMB':(x,y)}

    f = np.zeros(self.weights_len)
    crates = [[xi,yi] for xi in range(1,16) for yi in range(1,16) if (arena[xi,yi] == 1)]
    
    me = (x,y)                          # current position
    me_next = dict_actions[action]      # next position after performing action
    
    self.me = me # only debug
    
    # check if the next step brings us closer to a coin
    coins = np.array(game_state['coins'])
    if len(coins) != 0:
        dist = np.min(distance(me, coins))
        dist_next = np.min(distance(me_next, coins))
        f[0] = .2*np.sign(dist-dist_next)
        f[8] = 1./max(dist,1)   # avoid division by zero
            
    # when in doubt: rather do something than nothing
    if action=='WAIT':
        f[1] = -.2
    elif action=='BOMB':
        f[1] = +.1
    
    #try not to stay in one place
    if me_next in list(coordinate_history):
        f[2] = -1*(1. - list(coordinate_history)[::-1].index(me_next)/len(coordinate_history))
    
    # check if the next step brings us closer to a crate
    if len(crates) != 0:
        dist = np.min(distance(me, crates))
        dist_next = np.min(distance(me_next, crates))
        f[3] = .2*np.sign(dist-dist_next)   
    
    # it might be a good idea to drop a bomb if many crates would be destroyed
    if action == 'BOMB' and len(crates) != 0:
        expl_x,expl_y = get_explosion_xys(me, arena, s.bomb_power).T    # get coordinates of explosion for bomb dropped at (me)
        f[4] = 5*.02 * len(np.where(arena[expl_x,expl_y]==1)[0])           # for each crate that would be destroyed, reward 0.2
        if len(others) == 0:
            f[4]*=10
    # RUN FORREST RUN!
    #if bomb_map [me_next] == 0:
    #    f[5]=-.5
                
    
    # check if possible to escape when dropping a bomb
    if action=='BOMB':
        f[5] = can_run_or_hide(self, state, me, action) - 0.5
    else:
        f[5] = can_run_or_hide(self, state, me_next) - 0.5
    #self.logger.debug(f'me={me},me_next={me_next}')
    
    # check if there is an opponent in our vicinity
    if len(others) > 0:
        dist_opp = np.min(distance(me,np.array(others)))
        if dist_opp < 5:
            f[6] = 1-0.2*dist_opp
    
    
    if action == 'BOMB' and len(others) > 0:
        for p in list(dict_actions.values()):
            if p in others:
                f[7] += 0.5
        for p in [(min(x+2,16),y), (max(x-2,0),y),(x,min(y+2,16)),(x,max(y-2,0)),(x+1,y+1),(x+1,y-1),(x-1,y+1),(x+1,y-1)]:
            if p in others:
                f[7] += 0.25
    
    ## higher order correclations:
    f_two_point=[]
    for i in range(self.weights_len):
        for j in range(i):
            f_two_point.append(0.1*f[i]*f[j])
    f = np.append(f,f_two_point)

    self.logger.debug(f'f={f}')
    return f


def get_Q(self, state, action):
    """
    returns the state-action value Q given a state and a (proposed) action
    using the coefficient functions computed in funcs(),
    here only the scalar product with the learned weights self.weights is computed    
    """
    return np.dot(self.weights, funcs(self, state, action))

        
def act(self):
    """Called each game step to determine the agent's next action.

    - prepare 'state' variable [game_state, coordinate_history, bomb_map]
    - determine valid actions
    - use \epsilon-greedy policy to determine next action
          (by calculating Q(s,a) for every valid action a)
    """
    self.logger.info('Picking action according to rule set')

    # Gather information about the game state
    arena = self.game_state['arena']                                # map
    x, y, _, bombs_left, _ = self.game_state['self']                # our coordinates
    bomb_xys = [(x,y) for (x,y,t) in self.game_state['bombs']]      # bomb coordinates
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]   # other agents' coordinates

    # compute a 'bomb map' indicating when and where there will be an explosion on each tile
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in self.game_state['bombs']:
        for (i,j) in get_explosion_xys((xb,yb),arena):
            bomb_map[i,j] = min(bomb_map[i,j], t)
            
    # bundle in state variable
    self.coordinate_history.append((x,y))
    self.state = [(self.game_state).copy(), list(self.coordinate_history), bomb_map.copy()] 

    
    # determine valid actions
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (self.game_state['explosions'][d] <= 1) and
            (bomb_map[d] > 0) and
            (not d in others) and
            (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x-1,y) in valid_tiles: valid_actions.append('LEFT')
    if (x+1,y) in valid_tiles: valid_actions.append('RIGHT')
    if (x,y-1) in valid_tiles: valid_actions.append('UP')
    if (x,y+1) in valid_tiles: valid_actions.append('DOWN')
    if (x,y)   in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')
    
    if len(valid_actions) == 0:
        return # there's not much to think about
    
    ### epsilon-greedy action selection:
    if np.random.rand()>self.epsilon:       # with prob. 1-epsilon choose action that maximizes Q
        Qs = []
        for action in valid_actions:
            Qs.append(get_Q(self, self.state, action))
            self.logger.debug(f'Q(s,{action})={Qs[-1]}')
        self.next_action =  valid_actions[np.argmax(Qs)]
    else:                                   # with prob. epsilon choose random action
        self.next_action = np.random.choice(valid_actions)
    


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    - calculate rewards based on observed game events
    - store all relevant quantities for learning (i.e. state, action and reward)
    - if SARSA update after every step is enabled (self.update == move): perform SARSA update
    
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')

    self.logger.debug(f'Game event(s): {self.events}')    
    
    r = 0
    
    for event in self.events:
        if event in [0,1,2,3]:          # MOVED left, right, up or down
            r -= 0.1
        elif event == 4:                # WAITED
            r -= 0.2
        elif event == 5:                # INTERRUPTED
            self.logger.debug('-------- ERROR: PROCESS INTERRUPTED ----------- we were too slow')
        elif event == 6:                # INVALID_ACTION
            invalid_action = True
            x,y = self.me
            for p in [(min(x+2,16),y), (max(x-2,0),y),(x,min(y+2,16)),(x,max(y-2,0)),(x+1,y+1),(x+1,y-1),(x-1,y+1),(x-1,y-1)]:
                if p in [(x,y) for (x,y,n,b,s) in self.state[0]['others']]:
                    self.logger.debug('invalid action: tried to move on a tile at the same time as an opponent')
                    invalid_action = False
                    break
            if invalid_action:
                self.logger.debug('-------- ERROR: INVALID ACTION ----------- this should not happen!')
                self.logger.debug(f'{self.last_map}\n{self.last_bmap}\nme = {self.me}')
        elif event == 7:                # BOMB_DROPPED
            r += .0
        elif event == 8:                # BOMB_EXPLODED
            r += 0
        elif event == 9:                # CRATE_DESTROYED
            r += .5
        elif event == 10:               # COIN_FOUND
            r += .5
        elif event == 11:               # COIN_COLLECTED
            r += s.reward_coin*2
        elif event == 12:               # KILLED_OPPONENT
            r += s.reward_kill*2
        elif event == 13:               # KILLED_SELF
            r -= 5
        elif event == 14:               # GOT_KILLED
            r -= 10
        elif event == 15:               # OPPONENT_ELIMINATED
            r += 0
        elif event == 16:               # SURVIVED_ROUND
            r += 0

    self.reward_sum += r        # only for monitoring
    
    self.reward_hist.append(r)                  # save reward to queue for learning
    self.action_hist.append(self.next_action)   # save action to queue for learning
    self.state_hist.append(self.state)          # save state  to queue for learning
    
    self.logger.debug(f'Reward r = {r}')

    if self.update=='move':
        ### SARSA
        delta = r + self.gamma*get_Q(self.state_hist[-1], self.action_hist[-1]) - get_Q(self.state_hist[-2], self.action_hist[-2]) 
        self.logger.debug(f'Difference delta = {delta}')
        
        f = funcs(self.state_hist[-2], self.action_hist[-2])
        
        self.weights += self.alpha * f * delta
        with open(self.out_file_weight,'a') as fd:
            wr = csv.writer(fd)
            wr.writerow(self.weights)
        
        self.logger.debug(self.weights)
    
def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    - perform n-step SARSA learning (if self.update is an integer which in this case is interpreted as n)
    - save updated weights to disk
    - save data for monitoring and debugging
    - update episode counter and anneal epsilon
    - reset queues and variables for next episode
    
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step of episode {self.episode_nr}')
    reward_update(self)    
    
    if len(self.state_hist) != len(self.action_hist) or len(self.action_hist) != len(self.reward_hist):
        self.logger.debug('ERROR in end_of_episode: lengths dont match:')
        self.logger.debug(f'len(self.state_hist)= {len(self.state_hist)}; len(self.action_hist)= {len(self.action_hist)}; len(self.reward_hist)= {len(self.reward_hist)}')
    ### n-step SARSA
    if type(self.update)==np.int:       # if self.update is an integer, do n-step updates after every episode
        for t in range(len(self.reward_hist)-1):
            n = self.update
            while t + n > len(self.reward_hist)-1: n -= 1
            self.logger.debug(f't={t}; n={n}; len(self.reward_hist)-1= {len(self.reward_hist)-1}')
            delta = np.sum([self.gamma**(ti-t-1) * self.reward_hist[ti] for ti in np.arange(t+1,t+n+1,1)])
            delta += self.gamma**n * get_Q(self, self.state_hist[t+n], self.action_hist[t+n]) - get_Q(self, self.state_hist[t], self.action_hist[t]) 
            upd = self.alpha * funcs(self, self.state_hist[t], self.action_hist[t]) * delta
            self.logger.debug(f'Difference delta(t={t}) = {delta}, weights + = {upd}')
            self.weights += upd
            with open(self.out_file_weight,'a') as fd:
                wr = csv.writer(fd)
                wr.writerow(self.weights)
    
    # only debug       
#    np.save('state_hist', self.state_hist)
#    np.save('action_hist', self.action_hist)
#    np.save('reward_hist', self.reward_hist)

    # save weights and monitoring data
    np.save(self.file_weights, self.weights)
    with open(self.out_file_reward,'a') as fd:
        wr = csv.writer(fd)
        wr.writerow([self.reward_sum, np.mean(self.reward_hist)])
    

    self.episode_nr += 1
    
    # anneal epsilon
    self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.*self.episode_nr / self.eps_decay)
    self.logger.debug(f'after episode {self.episode_nr}: epsilon = {self.epsilon}')
    
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.state_hist = deque([],s.max_steps)     # state history, needed for SARSA learning
    self.action_hist = deque([],s.max_steps)    # action history, -------"------------
    self.reward_sum = 0.                        # sum of rewards per episode, for monitoring
    self.reward_hist = deque([],s.max_steps)    # list of all rewards during current episode
    self.logger.debug('juhuu, no errors!')

    # show learning progress during long sessions
    if self.episode_nr%100 == 0:
        print(f'Episode {self.episode_nr} complete!')