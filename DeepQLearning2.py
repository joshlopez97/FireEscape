import numpy as np
import os
import sys
import time
import json
import random
import tensorflow as tf
import pickle

try:
    from malmo import MalmoPython
except:
    import MalmoPython

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

#--------------------------------------- Main ---------------------------------------
mission_file = 'map4.xml'

#DQN init ---------------------------------------------------------------------------
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,1764],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([1764,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

#create lists to contain total rewards and steps per episode
rList = []
jList = []

#DQN parameters  
eps = 0.1
y = 0.99
num_episodes = 10 
iterationsWithNoRandom = 0
eps_deg = eps/(num_episodes - iterationsWithNoRandom)
#DQN init end ----------------------------------------------------------------------

#action list = north, south, west, east
#this calculation is reliant on knowing the grid is 21x21
action_trans = [(-21,'movenorth 1'), (21, 'movesouth 1'), (-1, 'movewest 1'), (1, 'moveeast 1')] 
#Printing Variables
actionlist = {-21: 'movenorth 1', 21: 'movesouth 1', -1: 'movewest 1', 1: 'moveeast 1'}
moveList = []
#this is tuned for 2x7 board2
optimalRes = ['movenorth 1', 'movenorth 1', 'movenorth 1', 'movenorth 1', 'movenorth 1', 'movenorth 1', 'movewest 1', 'movewest 1']
errorLog = []

#init the tensorflow session
with tf.Session() as sess:
    sess.run(init)

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

    if agent_host.receivedArgument("test"):
        num_repeats = 1
    else:
        num_repeats = num_episodes

    #Lets us use the various minecraft cheats available
    agent_host.sendCommand("chat /gamerule naturalRegeneration false")

    #map file selection
    f = open(mission_file, "r") 
    missionXML = f.read()
    my_mission = MalmoPython.MissionSpec(missionXML, True)

    for i in range(num_repeats):
        count = i
        print()
        print('Repeat %d of %d' % ( i+1, num_repeats ))

        #video recording
        #put a string into MissionRecordSpec("string") as file name
        my_mission_record = MalmoPython.MissionRecordSpec()  
        my_mission.requestVideo(1200,720)
        my_mission.setViewpoint(1)

        #comment this out if not recording anything
        # my_mission_record.recordMP4(30, 2000000) 

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

        grid = load_grid(world_state)
        start, end = find_start_end(grid) #start, end = gridIndex

        #DQN start-------------------------------------------------------------------------------
        #Reset environment and get first new observation
        s = start
        rAll = 0.0
        done = False 
        j = 0
        fireCount = 0

        #The Q-Table learning algorithm
        while j < 99:
            time.sleep(0)  #0.35 will cause the 3 fire steps to kill the agent 
            j+=1

            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(1764)[s:s+1]})

            #prob of eps to choose random action
            if (np.random.rand(1)<eps) and (eps>0):
                a[0] = np.random.randint(0, len(action_trans)-1)

            #step(a[0]) = Get new state and reward from environment
            agent_host.sendCommand(action_trans[a[0]][1])  #gets action of a
            s1 = s + action_trans[a[0]][0] #gets index of a

            #used to send commands etc
            s1Trans = s1   #s1 translated back to fire=0 states
            sTrans = s     #s translated back to fire=0 states
            if s1 >= 1323:
                s1Trans = s1 - 1323
                sTrans = s - 1323
            elif s1 >= 882:
                s1Trans = s1 - 882
                sTrans = s - 882
            elif s1 >= 441:
                s1Trans = s1 - 441
                sTrans = s - 441

            #calculating immediate reward
            curPath = dijkstra_shortest_path(grid, s1Trans, end)
            if grid[s1Trans] == 'air':
                r = -999
                done = True
            elif grid[s1Trans] == 'netherrack':
                r = -(len(curPath)-1)
                #never stepped on fire (full health)
                if fireCount == 0: #0-440
                    r += -2.5
                    fireCount += 1
                    s1 += 441
                #stepped on fire once already (half health)
                elif fireCount == 1: #441-881
                    r += -4
                    fireCount += 1
                    s1 += 441
                #stepped on fire twice already (next touch is death)
                elif fireCount == 2: #882-1322  
                    r += -999
                    fireCount += 1
                    done = True
            elif grid[s1Trans] == 'redstone_block':
                r = 5
                done = True
            else:
                r = -(len(curPath)-1)

            #testing fire state moves
            #chatState = "Run " + str(count) + ": state = " + str(s) + " | s1 = " + str(s1) + " | fire = " + str(fireCount)
            #print(chatState)

            #Update Q-Table with new knowledge
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(1764)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(1764)[s:s+1],nextQ:targetQ})
            rAll += r

            #for printing
            s_diff = sTrans - s1Trans
            moveList.append(actionlist[s_diff])
            
            #increment s to s1
            s = s1

            print("Run %d | Reward = %d" %(count, r))

            if done == True:
                #Reduce chance of random action as we train the model.
                eps = eps-eps_deg

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
                    errorLog.append((count, errorCount))

                moveList.clear()
                break

        jList.append(j)
        rList.append(rAll)
        print("rList for %d: %d" %(count, rList[count]))
        print("jList for %d: %d" %(count, jList[count]))
    
    #dump errorLog into 
    statFileName = "DeepQLearning2Stats/DeepQLearning2_" + mission_file[0:4] + "_stats.dat"
    rewardFileName = "DeepQLearning2Stats/DeepQLearning2_" + mission_file[0:4] + "_rewards.dat"
    moveFileName = "DeepQLearning2Stats/DeepQLearning2_" + mission_file[0:4] + "_moves.dat"
    np.savetxt(statFileName, errorLog)
    np.savetxt(rewardFileName, rList)
    np.savetxt(moveFileName, jList)
        
