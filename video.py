import numpy as np
import os
import sys
import time
import json
import random

try:
    from malmo import MalmoPython
except:
    import MalmoPython

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

canvas = None
root = None

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
    start = grid.index("emerald_block")
    end = grid.index("redstone_block")
    return (start, end)

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

    direction = [21, -1, -21, 1]
    vertexdict = dict()
    unvisited = []
    for i in range(len(grid_obs)):
        if grid_obs[i] != 'air': #<----------- Add things to avoid here
            vertexdict[i] = [1, 999, -999]  #key = index, value = (cost, shortest dist from start, prev vert)
            unvisited.append(i)  #add to unvisited list

    #set source vertex cost and shortest_dist_from_start to 0
    if source in vertexdict:
        vertexdict[source][0] = 0
        vertexdict[source][1] = 0
    else:
        return np.zeros(99)

    while len(unvisited) != 0:
        #find curVert - lowest shortest dist vertex
        lowestDist = float('inf')
        curVert = None
        for i in unvisited:
            if vertexdict[i][1] < lowestDist:
                curVert = i
                lowestDist = vertexdict[i][1]

        #examine neighbors of curVert
        for i in direction:
            adjVert = curVert + i
            if adjVert in unvisited:
                #newcost = (cost of adjVert) + (shortest dist from curVert)
                newCost = vertexdict[adjVert][0] + vertexdict[curVert][1]
                if newCost < vertexdict[adjVert][1]:
                    vertexdict[adjVert][1] = newCost
                    vertexdict[adjVert][2] = curVert
        unvisited.remove(curVert)

    backtrack = dest
    path_list = []
    path_list.append(dest)
    while backtrack != source:
        path_list.insert(0, vertexdict[backtrack][2])
        backtrack = vertexdict[backtrack][2]
    return path_list

def drawQ(fire_cells, q_table, canvas, root, start, curr_x=None, curr_y=None, width = 3, height = 7):
    scale = 40
    world_x = width
    world_y = height
    fire = list()
    #print(curr_x, curr_y)
    #fire = [(0,1),(1,1),(2,1),(3,1), (1,3),(2,3),(3,3),(4,3)]
    if canvas is None or root is None:
        root = tk.Tk()
        root.wm_title("Q-table")
        canvas = tk.Canvas(root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
        canvas.grid()

        windowWidth = root.winfo_reqwidth()
        windowHeight = root.winfo_reqheight()

        positionRight = int(root.winfo_screenwidth()/7 - windowWidth)
        positionDown = int(root.winfo_screenheight()/50)

        # Positions the window in the center of the page.
        root.geometry("+{}+{}".format(positionRight, positionDown))
        root.update()

    canvas.delete("all")
    action_inset = 0.1
    action_radius = 0.1
    curr_radius = 0.2
    action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]

    # (NSWE to match action order)
    min_value =  -20
    max_value =  10

    for x in range(world_x):
        for y in range(world_y):
            index = start + y * 21 + x
            canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
            if ((x,y) in fire_cells):
                canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#ffb6c1")

            all = 1
            for action in range(4):
                if (q_table[index, action] != 0.0):
                    all = 0
            if (all == 1):
                continue
            for action in range(4):
                if not (0 <= index < len(q_table)):
                    continue
                max_value = max(q_table[index])
                min_value = max_value - 5
                value = q_table[index,action]
                color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                color_string = '#%02x%02x%02x' % (255-color, color, 0)
                canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                         (y + action_positions[action][1] - action_radius ) *scale,
                                         (x + action_positions[action][0] + action_radius ) *scale,
                                         (y + action_positions[action][1] + action_radius ) *scale,
                                         outline=color_string, fill=color_string )
    if curr_x is not None and curr_y is not None:
        if 0 <= curr_x < world_x and 0 <= curr_y < world_y:
            canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale,
                                     (curr_y + 0.5 - curr_radius ) * scale,
                                     (curr_x + 0.5 + curr_radius ) * scale,
                                     (curr_y + 0.5 + curr_radius ) * scale,
                                     outline="#fff", fill="#fff" )
    root.update()
    return canvas, root

#--------------------------------------- Main ---------------------------------------

#action list = north, south, west, east
#this calculation is reliant on knowing the grid is 21x21
action_trans = [(-21,'movenorth 1'), (21, 'movesouth 1'), (-1, 'movewest 1'), (1, 'moveeast 1') ]#, (-42, 'jumpnorth 2'), (42, 'jumpsouth 2'), (-2, 'jumpwest 2'), (2, 'jumpeast 2')]

action_cost = [0,0,0,0]
success = 0
#Q-table initializer
Q = np.zeros([441, len(action_trans)]) #441 = len(grid)
# for i in range(len(Q)):
#     Q[i] = [0,0,0,0,-2,-2,-2,-2]

#Set learning parameters
eps = 0.1
lr = .9
y = .9
num_episodes = 1000
iterationsWithNoRandom = 250   #<-----------------------------------------iteration randomness stop
eps_deg = eps/(num_episodes - iterationsWithNoRandom)

#create lists to contain total rewards and steps per episode
rList = []

#drawing path
canvas = None
root = None


#Printing and Error Log
optimalRes = ['movenorth 1', 'movenorth 1', 'movenorth 1', 'movenorth 1', 'movenorth 1', 'movenorth 1', 'movewest 1', 'movewest 1','movenorth 1']
errorLog = []
actionlist = {-21: 'movenorth 1', 21: 'movesouth 1', -1: 'movewest 1', 1: 'moveeast 1'} # -42: 'jumpnorth 2', 42: 'jumpsouth 2', -2: 'jumpwest 2', 2: 'jumpeast 2'}
moveList = []
jList = []

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


mission_file = "./map2.xml"
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = num_episodes

for i in range(num_repeats):
    print()
    print('Repeat %d of %d' % ( i+1, num_repeats ))
    count = i

    #setup mission to start
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(1500, 1200)
    agent_host.sendCommand("look -1")
    my_mission.setViewpoint(1)
    # Attempt to start a mission:
    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "%s-%d" % ('Moshe', i))
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

    #Q-learning
    grid = load_grid(world_state)
    start, end = find_start_end(grid) #start, end = gridIndex
    fire_cells = [ ((i-start) % 21, (i-start) // 21)  for i in range(len(grid)) if grid[i] == 'netherrack']

    #Reset environment and get first new observation
    s = start
    rAll = 0.0
    done = False #done
    j = 0
    fireCount = 0
    canvas, root = drawQ(fire_cells, Q, canvas, root, start, (s-start) % 21, (s-start) // 21 , width = 3, height = 8)
    #The Q-Table learning algorithm
    while j < 99:
        time.sleep(0.5)  #<----- adjust sleep

        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        #a = np.argmax(Q[s,:] + np.random.randn(1,len(action_trans)) * (1./(i+1)))

        rng = np.random.randint(1, 100)
        if rng>=1 and rng<=(100*eps): #P(eps)
            a = np.random.randint(0, len(action_trans)-1)
        else:
            a = np.argmax(Q[s,:])

        #Get new state and reward from environment
        s1 = s + action_trans[a][0] #gets index of a
        canvas, root = drawQ(fire_cells, Q, canvas, root, start, (s1-start) % 21, (s1-start) // 21 , width = 3, height = 8)
        #calculating reward <------------- variable
        curPath = dijkstra_shortest_path(grid, s1, end)
        if (agent_host.getWorldState().number_of_observations_since_last_state == 0):
            done = True

        if grid[s1] == 'air':
            r = -99
            done = True
        elif grid[s1] == 'netherrack':
            r = (-1.0*(len(curPath)-1))
            r = r - 2.5

        elif grid[s1] == 'redstone_block':
            r = 5
            done = True
        else:
            r = -1*(len(curPath)-1)

        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r

        #move agent
        command = action_trans[a][1]
        agent_host.sendCommand(command)  #gets action of a

        #calculating diff to print
        s_diff = s - s1
        moveList.append(actionlist[s_diff])

        #increment s
        s = s1
        if done == True:
            if (count%10) == 0:
                print()
                print("Report for %d: " % count)
                print("Path length found: ", len(moveList))
                print("Move list found: ", moveList)
                print()

            #error Calculation
            errorCount = 0
            for i in range(len(moveList)):
                if i > (len(optimalRes)-1):
                    errorCount += 1
                elif moveList[i] != optimalRes[i]:
                    errorCount += 1
            if len(moveList) < len(optimalRes):
                lengthDiff = len(optimalRes) - len(moveList)
                errorCount += lengthDiff
            if moveList == optimalRes:
                success += 1
            errorLog.append((count, errorCount))

            moveList.clear()
            break
    for i in [220,221,222,241,242,243,262,264,264,283,284,285,304,305,306,325,326,327,346,347,348,367,368,369]:
        print(f"{(i-start) % 21}:{(i-start) // 21} {Q[i]}")
    print("Result for Mission " + str(count+1))
    print("total reward: ", rAll)

    rList.append(rAll)
    jList.append(j)
    print("Number of moves per run: " + str(jList[count]))
    print("%Successful runs: " +  str(1.0*success/num_episodes))


#dump errorLog into
statFileName = "QLearning_" + mission_file[0:4] + "_stats.dat"
rewardFileName = "QLearning_" + mission_file[0:4] + "_rewards.dat"
np.savetxt(statFileName, errorLog)
np.savetxt(rewardFileName, rList)
