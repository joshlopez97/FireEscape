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

GAP_BLOCK = 'fire'

# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

def GetMissionXML(seed, gp, size=10):
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Hello world!</Summary>
              </About>

            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                  <DrawingDecorator>
                    <DrawSphere x="-27" y="70" z="0" radius="30" type="air"/>
                  </DrawingDecorator>
                   <MazeDecorator>
                    <Seed>'''+str(seed)+'''</Seed>
                    <SizeAndPosition width="''' + str(size) + '''" length="''' + str(size) + '''" height="1" xOrigin="-32" yOrigin="70" zOrigin="-5"/>
                    <FloorBlock type="diamond_block"/>
                    <StartBlock type="air" fixedToEdge="true"/>
                    <EndBlock type="air" fixedToEdge="true"/>
                    <PathBlock type="air"/>
                    <GapBlock type="fire"/>
                    <GapProbability>''' + str(0.2)+ '''</GapProbability>
                    <AllowDiagonalMovement>false</AllowDiagonalMovement>
                  </MazeDecorator>                     
                  <MazeDecorator>
                    <Seed>'''+str(seed)+'''</Seed>
                    <SizeAndPosition width="''' + str(size) + '''" length="''' + str(size) + '''" height="1" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
                    <StartBlock type="emerald_block" fixedToEdge="true"/>
                    <EndBlock type="redstone_block" fixedToEdge="true"/>
                    <PathBlock type="diamond_block"/>
                    <GapBlock type="gold_block"/>
                    <FloorBlock type="air" height="0"/>
                    <GapProbability>''' + str(0.2)+ '''</GapProbability>
                    <AllowDiagonalMovement>false</AllowDiagonalMovement>
                  </MazeDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="10000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>CS175AwesomeMazeBot</Name>
                <AgentStart>
                    <Placement x="0.5" y="56.0" z="0.5" yaw="0"/>
                </AgentStart>
                <AgentHandlers>
                    <DiscreteMovementCommands/>
                    <AgentQuitFromTouchingBlockType>
                        <Block type="redstone_block"/>
                    </AgentQuitFromTouchingBlockType>
                    <ObservationFromGrid>
                      <Grid name="floorAll">
                        <min x="-10" y="-1" z="-10"/>
                        <max x="10" y="-1" z="10"/>
                      </Grid>
                  </ObservationFromGrid>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''


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

num_repeats = 1

for i in range(num_repeats):
    size = 5
    print("Size of maze:", size)
    my_mission = MalmoPython.MissionSpec(GetMissionXML("0", 0.4, size), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)
    # Attempt to start a mission:
    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "%s-%d" % ('Moshe', i) )
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
        #sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission", (i+1), "running.")

    grid = load_grid(world_state)
    start, end = find_start_end(grid) # implement this
    path = dijkstra_shortest_path(grid, start, end)  # implement this
    action_list = extract_action_list_from_path(path)
    print("Output (start,end)", (i+1), ":", (start,end))
    print("Output (path length)", (i+1), ":", len(path))
    print("Output (actions)", (i+1), ":", action_list)
    # Loop until mission ends:
    action_index = 0
    while world_state.is_mission_running:
        #sys.stdout.write(".")
        time.sleep(0.1)

        # Sending the next commend from the action list -- found using the Dijkstra algo.
        if action_index >= len(action_list):
            print("Error:", "out of actions, but mission has not ended!")
            time.sleep(2)
        else:
            agent_host.sendCommand(action_list[action_index])
        action_index += 1
        if len(action_list) == action_index:
            # Need to wait few seconds to let the world state realise I'm in end block.
            # Another option could be just to add no move actions -- I thought sleep is more elegant.
            time.sleep(2)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission", (i+1), "ended")
    # Mission has ended.
