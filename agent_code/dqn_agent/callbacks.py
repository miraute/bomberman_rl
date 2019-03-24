################################# ANN version ##################################
import numpy as np
from collections import deque
import csv
import pickle


from settings import s

import random
import torch
import torch.optim as optim

from ANN import Memory, DQN, optimize_model

# set device for computation: 
use_gpu = True    # if True: use gpu (if available), else: use cpu
device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

# saves data (here intended to use for the dict returned by DQN.state_dict() ) using pickle
def save_weights(path, weights):
    with open(path, "wb") as f:
        pickle.dump(weights,f)

# loads the data saved by wave_weights()
def load_weights(path):
    with open(path,"rb") as f:
        weights = pickle.load(f)
    return weights


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    - set up 'global' variables / hyper-params and save them to self-object
    - initialize networks
    - 
    """
    self.logger.debug('Successfully entered setup code')
    
    #import os    # find out current path and working directory
    #self.logger.debug(f'path: {os.path.dirname(os.path.realpath(__file__))}, WD: {os.getcwd()}')
    
    np.random.seed()
    
    # hyper parameters
    self.batch_size = 1024      # size for mini-batches # 128
    self.gamma = 0.999          # discount factor
    self.eps_start = 0.01       # start value of epsilon (for epsilon-greedy policy)   # 0.9
    self.eps_end = 0.01         # end value of epsilon; epsilon is annealed as:        # 0.05
    self.eps_decay = 200        # eps = eps_end + (eps_start - eps_end) * exp(-1.*step_nr / eps_decay)
    self.step_nr = 0            # counts the steps from the beginning of learning
    self.episode_nr = 0         # counts the episodes -------- " ---------------
    self.target_update = 10     # target net is updated after every target_update episodes
    
    self.state_hist = deque([],2)   # state history, needed for making last 2 states available in reward_update()
    self.action_hist = deque([],2)  # action history, -----------"----------- actions -------- " -----------


    # load weights if possible 
    self.file_weights = 'weights.pkl'
    try:
        weights = torch.load('weights.pt')              # load trained weights
        self.logger.debug('Weight file successfully loaded!')
    except:
        weights = None
        self.logger.debug('ERROR: No weight file found!')
        pass
    try:
        optimizer_state = torch.load('opt.pt')
        self.logger.debug('Optimizer_state file successfully loaded!')
    except:
        optimizer_state = None
        self.logger.debug('ERROR: No optimizer_state file found!')
        pass
    
    self.pnet = DQN(s.cols, s.rows).to(device)          # initialize policy network...
    if weights:
        self.pnet.load_state_dict(weights)              #  ... with trained weights (if they already exist)
    self.tnet = DQN(s.cols, s.rows).to(device)          # initialize target network
    self.tnet.load_state_dict(self.pnet.state_dict())   #  ... with the same weights as the policy net
    self.tnet.eval()                                    # and set it to evaluation mode
    
    self.optimizer = optim.RMSprop(self.pnet.parameters())      # setup optimizer
    if optimizer_state:
        self.optimizer.load_state_dict(optimizer_state)
    self.device = device                                # store device for use in ANN file
    self.memory = Memory(10000)                         # create instance of Memory for replay memory

    # monitoring
    self.reward_hist = deque([],s.max_steps)    # list of all rewards during current episode
    self.out_file_reward = 'out.csv'            # csv-file where reward_sum is appended to after each episode
    
# returns all tiles that are hit by an explosion starting at start (map supplies positions of walls)
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
        return expl

# state preparation   
def get_state(self):
    # prepare map in state[0,:,:]
    state = np.ones((2,17,17))*5                    # init array
    state[0,:,:] = self.game_state['arena']         # fill with values from arena
    if len(self.game_state['coins'])>0:
        coinsx, coinsy = np.array(self.game_state['coins']).T
        state[0,coinsx, coinsy] = 2                 # set positions of coins
    if len(self.game_state['bombs']) > 0:
        bombs_x,bombs_y = np.array([[x,y] for (x,y,t) in self.game_state['bombs']]).T
        state[0,bombs_x,bombs_y] = -2               # set positions of bombs
    if len(self.game_state['others']) > 0:
        others_x,others_y = np.array([[x,y] for (x,y,n,b,s) in self.game_state['others']]).T
        state[0,others_x,others_y] = -3             # set positions of opponents
    state[0][self.game_state['self'][:2]] = 4       # set position of agent
    
    # preapre bomb map in state[1,:,:]
    for xb,yb,t in self.game_state['bombs']:
        for (i,j) in get_explosion_xys((xb,yb),self.game_state['arena'],s.bomb_power):
                state[1,i,j] = min(state[1,i,j], t)
    explosions = -1*self.game_state['explosions']
    explosions[explosions==0] = 5
    state[1,:,:] = np.minimum(state[1,:,:],explosions)
    
    # convert to tensor, manage dimensions, etc..
    state_t = np.ascontiguousarray(state, dtype=np.float32)
    state_t = torch.from_numpy(state_t)                   # convert to tensor
    return state_t.unsqueeze(0).to(device), state         # add a fake batch dimension and move to device


# old stack: torch.tensor with shape (1,8,h,w) or torch.from_numpy(np.array([np.nan]))
# state:     torch.tensor with shape (1,2,h,w)
# returns    torch.tensor with shape (1,8,h,w) where ret[0,0:2] = state and the rest moved 2 indeces to the back
#                or in the case that old_stack == nan: a stacked version of 4 times state
def stack(old_stack, state):
    if torch.any(torch.isnan(old_stack)):
        stacked = torch.zeros(tuple((8,state.shape[-2],state.shape[-1])),dtype=state.dtype,device=state.device)
        for i in range(0,8,2):
            stacked[i:i+2,:,:] = state
        return stacked.unsqueeze(0)
    for i in range(6,0,-2):
        old_stack[0,i:i+2,:,:]=old_stack[0,i-2:i,:,:]
    old_stack[0,0:2,:,:] = state
    return old_stack
    
    
### epsilon-greedy action selection
def get_action(self, state, valid_actions):
    epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.*self.step_nr / self.eps_decay)
    self.step_nr += 1
    if np.random.rand() > epsilon:          # with prob. 1-epsilon choose action that maximizes Q
        self.logger.debug('Chose action greedily')
        mask = [list(s.actions).index(a) for a in valid_actions]
        with torch.no_grad():
            ret = self.pnet(state)
        maxQ, max_idx = ret[0,mask].max(0)
        # monitoring
        with open('Q.csv','a') as fd:
            wr = csv.writer(fd)
            wr.writerow([maxQ])
        return valid_actions[max_idx]
    self.logger.debug('Random action was chosen')
    return random.choice(valid_actions)     # with prob. epsilon choose random action
        
    
def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    - compute valid actions
    - call get_state() to merge current state (from self.game_state) into 'state' variable
    - save current state to state_hist (deque with maxlen=2)
    - call select_action(map,valid_actions) and save it to action_hist (deque with maxlen=2)
    
    
    """
    self.logger.info('Picking action according to rule set')

    # computing state variable
    state, state_np = get_state(self)
    x, y, _, bombs_left, score = self.game_state['self']
    arena, bomb_map = state_np
    
    # determine valid actions
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if (((arena[d] == 0) or (arena[d] == 2) ) and
            (self.game_state['explosions'][d] <= 1) and
            (bomb_map[d] > 0)):
            valid_tiles.append(d)
    if (x-1,y) in valid_tiles: valid_actions.append('LEFT')
    if (x+1,y) in valid_tiles: valid_actions.append('RIGHT')
    if (x,y-1) in valid_tiles: valid_actions.append('UP')
    if (x,y+1) in valid_tiles: valid_actions.append('DOWN')
    if (x,y)   in valid_tiles: valid_actions.append('WAIT')
    if (bombs_left > 0): valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')
    
    if len(valid_actions) == 0:
        return # we're fucked -> can only happen in last step of an episode
    
    # prepare state by stacking the the current state on top of 3 past states
    old_stack = self.state_hist[-1] if len(self.state_hist) > 0 else torch.from_numpy(np.array([np.nan]))
    stacked_state = stack(old_stack,state)

    # decide next action
    action= get_action(self, stacked_state, valid_actions)
    self.next_action = action
    
    # save state and action such that they are available in reward_update for learning
    self.action_hist.append(action)
    self.state_hist.append(stacked_state)

   
def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    - compute reward
    - memory.save(state, action, next_state, reward)
        where state = state_hist[-2], action = action_hist[-2], next_state = state_hist[-1]
    - call optimize_model()
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')

    self.logger.debug(f'Game event(s): {self.events}')    
    
    # calculate reward
    r = 0.
    for event in self.events:
        if event in [0,1,2,3]:          # MOVED left, right, up or down
            r -= 0.1
        elif event == 4:                # WAITED
            r -= 0.2
        elif event == 5:                # INTERRUPTED
            self.logger.debug('-------- ERROR: PROCESS INTERRUPTED ----------- we were too slow')
        elif event == 6:                # INVALID_ACTION
            invalid_action = True
            #x,y = np.argwhere(self.state_hist[-1].squeeze().numpy()[0,:,:]==4)[0] # get current position
            x, y = self.game_state['self'][:2]
            for p in [(min(x+2,16),y), (max(x-2,0),y),(x,min(y+2,16)),(x,max(y-2,0)),(x+1,y+1),(x+1,y-1),(x-1,y+1),(x-1,y-1)]:
                if p in [(x,y) for (x,y,n,b,s) in self.game_state['others']]:
                    self.logger.debug('invalid action: tried to move on a tile at the same time as an opponent')
                    invalid_action = False
                    break
            if invalid_action:
                self.logger.debug('-------- ERROR: INVALID ACTION ----------- this should not happen!')
        elif event == 7:                # BOMB_DROPPED
            r += .0
        elif event == 8:                # BOMB_EXPLODED
            r += 0
        elif event == 9:                # CRATE_DESTROYED
            r += .5
        elif event == 10:               # COIN_FOUND
            r += 1
        elif event == 11:               # COIN_COLLECTED
            r += s.reward_coin*2
        elif event == 12:               # KILLED_OPPONENT
            r += s.reward_kill*2
        elif event == 13:               # KILLED_SELF
            r -= 10
        elif event == 14:               # GOT_KILLED
            r -= 5
        elif event == 15:               # OPPONENT_ELIMINATED
            r += 0
        elif event == 16:               # SURVIVED_ROUND 
            r += 1                      #  (actually can never happen)

    # monitoring
    self.reward_hist.append(r)
    self.logger.debug(f'Reward r = {r}')
    
    # store (state, action, next_state, reward) to replay memory
    if len(self.action_hist)>1:
        action = torch.tensor([list(s.actions).index(self.action_hist[-2])], device=device)
        self.memory.save(self.state_hist[-2], action, self.state_hist[-1], torch.tensor([r], dtype=torch.float, device=device))
    
    # where the learning takes place:
    optimize_model(self)
    
    
def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    - memory.save(state, action, next_state=None, reward=0)
    - compute and save some meaningful indicator of success (e.g. mean reward per episode?)
    - every once in a while (hyper-param -> setup() ): update target network
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step of episode {self.episode_nr}')

    # due to the structure of the act() and reward_update() we are always one step behind with learning
    reward_update(self)
    
    # store (state, action, next_state=None, reward=0) to replay memory as this is the terminal state
    action = torch.tensor([list(s.actions).index(self.action_hist[-1])], device=device)
    self.memory.save(self.state_hist[-1], action, None, torch.tensor([0], dtype=torch.float, device=device))
    

    # save current weights 
    weights = self.pnet.state_dict()
    save_weights(self.file_weights, weights)
    torch.save(weights, 'weights.pt')
    torch.save(self.optimizer.state_dict(),'optimizer_state.pt')
    
    # every target_update episodes: update target network
    if self.episode_nr % self.target_update == 0:
        self.tnet.load_state_dict(weights)
    
    # increase episode counter
    self.episode_nr += 1

    # re-initialize queues
    self.state_hist = deque([],2)               # state history, needed for SARSA learning
    self.action_hist = deque([],2)              # action history, -------"------------
    
    # monitoring 
    with open(self.out_file_reward,'a') as fd:
        wr = csv.writer(fd)
        wr.writerow([np.sum(self.reward_hist), np.mean(self.reward_hist)])
    
    # reset monitoring
    self.reward_hist = deque([],s.max_steps)    # list of all rewards during current episode
    self.logger.debug('juhuu, no errors!')
    
    with open('Q.csv','a') as fd:               # mark end of episode in state action value file
        wr = csv.writer(fd)
        wr.writerow([None])

    # only debug
    if self.episode_nr%100 == 0:
        print(f'Episode {self.episode_nr} complete!')
