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

# Tutorial sample #7: The Maze Decorator

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
from priority_dict import priorityDictionary as PQ
from Q_learning import TabQAgent
GAP_BLOCK = 'fire'

# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

def load_grid(world_state):
    """
    Used the agent observation API to get a 21 X 21 grid box around the agent (the agent is in the middle).
    Args
        world_state:    <object>    current agent world state
    Returns
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)
    """
    while world_state.is_mission_running:
        #sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floorAll', 0)
            break
    return grid

def find_start_end(grid):
    """
    Finds the source and destination block indexes from the list.
    Args
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)
    Returns
        start: <int>   source block index in the list
        end:   <int>   destination block index in the list
    """
    return (grid.index("emerald_block"), grid.index("redstone_block"))


def extract_action_list_from_path(path_list):
    """
    Converts a block idx path to action list.
    Args
        path_list:  <list>  list of block idx from source block to dest block.
    Returns
        action_list: <list> list of string discrete action commands (e.g. ['movesouth 1', 'movewest 1', ...]
    """
    action_trans = {-21: 'movenorth 1', 21: 'movesouth 1', -1: 'movewest 1', 1: 'moveeast 1'}
    alist = []
    for i in range(len(path_list) - 1):
        curr_block, next_block = path_list[i:(i + 2)]
        alist.append(action_trans[next_block - curr_block])

    return alist


def get_neighbors(grid_obs, index):
    """
    Helper function returns indices of neighboring blocks that can be
    traveled to.
    """
    neighbors = []

    # Check left neighbor
    if index % 21 != 0 and grid_obs[index - 1] != GAP_BLOCK:
        neighbors.append(index - 1)

    # Check right neighbor
    if index % 21 != 20 and grid_obs[index + 1] != GAP_BLOCK:
        neighbors.append(index + 1)

    # Check top neighbor
    if index > 20 and grid_obs[index - 21] != GAP_BLOCK:
        neighbors.append(index - 21)

    # Check bottom neighbor
    if index < (len(grid_obs) - 21) and grid_obs[index + 21] != GAP_BLOCK:
        neighbors.append(index + 21)
    return neighbors

def is_solution(reward):
    return reward == 100


def find_min(distances, visited):
    """
    Helper function returns index of unvisited block with shortest distance
    from origin or None if it does not exist.
    """
    minIndex = None
    minDist = float('inf')
    for index, dist in distances.items():
        if dist < minDist and not visited[index]:
            minDist = dist
            minIndex = index
    return minIndex

def dijkstra_shortest_path(grid_obs, source, dest):
    """
    Finds the shortest path from source to destination on the map. It used the grid observation as the graph.
    See example on the Tutorial.pdf file for knowing which index should be north, south, west and east.
    Args
        grid_obs:   <list>  list of block types string representing the blocks on the map.
        source:     <int>   source block index.
        dest:       <int>   destination block index.
    Returns
        path_list:  <list>  block indexes representing a path from source (first element) to destination (last)
    """

    # INFINITY constant (non-air blocks)
    INF = float('inf')

    # Construct graph
    neighbors = dict()
    dist = dict()
    visited = dict()
    path_to = dict()
    graph = []
    for i, block in enumerate(grid_obs):
        if block != GAP_BLOCK:
            graph.append(i)
            neighbors[i] = get_neighbors(grid_obs, i)
            dist[i] = INF
            visited[i] = False
            path_to[i] = []

    dist[source] = 0

    while graph:
        block_i = find_min(dist, visited)

        if block_i == None or dist[block_i] == INF:
            break

        visited[block_i] = True

        graph.remove(block_i)

        for neighbor in neighbors[block_i]:
            new_distance = dist[block_i] + 1
            if not visited[neighbor] and (dist[neighbor] == INF or dist[neighbor] > new_distance):
                new_path = path_to[block_i] + [block_i]
                dist[neighbor] = new_distance
                path_to[neighbor] = new_path

    # Return shortest path to destination
    return path_to[dest] + [dest]


# Create default Malmo objects:
agent = TabQAgent()
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

num_repeats = 150

cumulative_rewards = []

size = 5
print("Size of maze:", size)

mission_file = './map1.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

for i in range(num_repeats):
    print('Repeat %d of %d' % ( i+1, num_repeats ))
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)
    # Attempt to start a mission:
    max_retries = 3
    # my_clients = MalmoPython.ClientPool()
    # my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission,my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission", (i+1), ":",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission", (i+1), "to start ",)
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission", (i+1), "running.")

    grid = load_grid(world_state)
    start, end = find_start_end(grid) # implement this

    # -- run the agent in the world -- #
    print("cumulative reward")
    cumulative_reward = agent.run(agent_host, start, end)
    print('Cumulative reward: %d' % cumulative_reward)
    if is_solution(cumulative_reward):
        print('Found solution')
        print('Done')
        break

    cumulative_rewards += [ cumulative_reward ]

    # -- clean up -- #
    time.sleep(0.5) # (let the Mod reset)
    print()
    print("Mission", (i+1), "ended")
    # Mission has ended.
print("Done.")

print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)
