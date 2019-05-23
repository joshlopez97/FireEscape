from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #6: Discrete movement, rewards, and learning

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import math
from collections import deque

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

missionEnd = -100

class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.epsilon = 0.1 # chance of taking a random action instead of the best
        self.alpha = 0.1 #learning rate
        self.gamma = 1.0 #discount rate

        self.size = 5
        self.start = 0 #start index
        self.dest = 0
        self.cur = 0

        self.cur_indexes = set()
        self.cur_indexes.add(self.start)

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

        self.action_cost = [0, 0, 0, 0]

        self.q_table = {}
        self.canvas = None
        self.root = None

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]

        self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * (reward + self.gamma * max(self.q_table[current_state]) - old_q)

    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]

        self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * ( reward - old_q )

    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        '''
        Return:
            a: an actions
            cur_reward: a reward for that action
        '''
        obs_text = world_state.observations[-1].text

        obs = json.loads(obs_text) # most recent observation

        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0

        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if current_s not in self.q_table:
            pos_indexes = self.possible_indexes1(self.cur)
            pos_indexes = [self.actions.index(i) for i in pos_indexes]
            self.q_table[current_s] = ([0] * len(self.actions))
            for i in range(len(self.actions)):
                if i not in pos_indexes:
                    self.q_table[current_s][i] = -1000

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, current_s )

        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            pos_indexes = self.possible_indexes2(self.cur)
            if len(pos_indexes) == 0:
                pos_indexes = self.possible_indexes1(self.cur)
            a = random.randint(0, len(pos_indexes) - 1)
            b = pos_indexes[a]
            a = self.actions.index(b)
            self.logger.info("Random action: %s" % b)

        else:
            l = list()
            pos_indexes = list()
            #print(self.q_table.items())

            m = max(self.q_table[current_s])
            for x in range(0, len(self.actions)):
                if self.q_table[current_s][x] == m:
                    l.append(x)

            l = [self.actions[i] for i in l]
            #print(l)
            pos_indexes = self.possible_indexes3(self.cur, l)
            if len(pos_indexes) != 0:
                a = random.randint(0, len(pos_indexes) - 1)
                b = pos_indexes[a]
                a = self.actions.index(b)
                self.logger.info("Taking q action: %s" % self.actions[a])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            action_trans = {'movenorth 1': -21, 'movesouth 1': 21, 'movewest 1': -1, 'moveeast 1': 1}
            self.cur = self.cur + action_trans[b]
            (self.cur_indexes).add(self.cur)
            #print("passed indexes", self.cur_indexes)
            agent_host.sendCommand(b)
            self.prev_s = current_s
            self.prev_a = a

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_r

    def possible_indexes1(self, index):
        indexes = {'N': self.actions[0], 'S': self.actions[1], 'W': self.actions[2], 'E': self.actions[3]}
        s = self.start % 21
        if (index <= self.start+20):
            del indexes['N']

        if (index % 21 == s):
            del indexes['W']

        if (index >= self.dest-20):
            del indexes['S']

        if (index % 21 == s + self.size-1):
            del indexes['E']

        return list(indexes.values())

    def possible_indexes2(self, index):
        indexes = {'N': self.actions[0], 'S': self.actions[1], 'W': self.actions[2], 'E': self.actions[3]}
        s = self.start % 21
        if (index <= self.start+20) or ((index-21) in self.cur_indexes):
            del indexes['N']

        if (index % 21 == s) or ((index-1) in self.cur_indexes):
            del indexes['W']

        if (index >= self.dest-20) or ((index+21) in self.cur_indexes):
            del indexes['S']

        if (index % 21 == s + self.size-1) or ((index+1) in self.cur_indexes):
            del indexes['E']

        return list(indexes.values())

    def possible_indexes3(self, index, l):
        indexes = {'N': self.actions[0], 'S': self.actions[1], 'W': self.actions[2], 'E': self.actions[3]}
        s = self.start % 21
        if (index <= self.start+20):
            if indexes['N'] in l:
                l.remove(indexes['N'])

        if (index % 21 == s):
            if indexes['W'] in l:
                l.remove(indexes['W'])

        if (index >= self.dest-20):
            if indexes['S'] in l:
                l.remove(indexes['S'])

        if (index % 21 == s + self.size-1):
            if indexes['E'] in l:
                l.remove(indexes['E'])

        return l

    def dijkstra_shortest_path_cost(self, cur_index, source, dest):
        dest_x, dest_y = (dest % 21, dest // 21)
        x, y = (cur_index % 21, cur_index // 21)
        return -(abs(x-dest_x) + abs(y-dest_y))

    def run(self, agent_host, start,dest ):
        """run the agent on the world"""

        self.S, self.A, self.R = deque(), deque(), deque()

        total_reward = 0
        cost = 0

        self.start = start
        self.dest = dest
        self.cur = start
        self.cur_indexes = set()
        self.cur_indexes.add(self.start)

        indexes_list = set()

        self.prev_s = None
        self.prev_a = None

        is_first_action = True

        # main loop:
        current_r = 0
        world_state = agent_host.getWorldState()

        while world_state.is_mission_running:
            current_r = 0 - cost

            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        ob = json.loads(world_state.observations[-1].text)['floorAll']
                        self.fire_cells = [ ((i-self.start) % 21, (i-self.start) // 21)  for i in range(len(ob)) if ob[i] == 'netherrack']
                        total_reward += self.act(world_state, agent_host, current_r)
                        cost = self.action_cost[self.prev_a]
                        current_r += self.dijkstra_shortest_path_cost(self.cur, self.start, self.dest)
                        #print("path cost:", self.dijkstra_shortest_path_cost(self.cur, self.start, self.dest))
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0 - cost:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()

                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        cost = self.action_cost[self.prev_a]
                        current_r += self.dijkstra_shortest_path_cost(self.cur, self.start, self.dest)
                        #print("path cost:", self.dijkstra_shortest_path_cost(self.cur, self.start, self.dest))
                        break
                    if not world_state.is_mission_running:
                        break


        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r
        if self.dest in self.cur_indexes:
            print("goal reached")

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )

        self.drawQ()

        return total_reward

    def drawQ( self, curr_x=None, curr_y=None ):
        scale = 40
        world_x = self.size
        world_y = self.size
        #fire = [(0,1),(1,1),(2,1),(3,1), (1,3),(2,3),(3,3),(4,3)]
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x,y)

                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                if ((x,y) in self.fire_cells):
                    self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#ffb6c1")

                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                    color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255-color, color, 0)
                    self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                             (y + action_positions[action][1] - action_radius ) *scale,
                                             (x + action_positions[action][0] + action_radius ) *scale,
                                             (y + action_positions[action][1] + action_radius ) *scale,
                                             outline=color_string, fill=color_string )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale,
                                     (curr_y + 0.5 - curr_radius ) * scale,
                                     (curr_x + 0.5 + curr_radius ) * scale,
                                     (curr_y + 0.5 + curr_radius ) * scale,
                                     outline="#fff", fill="#fff" )
        self.root.update()
