---
layout: default
title: Proposal
---

##Summary of the Project:
Our project will aim to train an agent to complete a challenge map in the least amount of time.  The challenge maps will consist of traps, enemies, obstacles, and platforming that the agent will be trained to navigate and avoid.  In particular, we are interested in reducing the completion time of the challenge map relative to the expected completion time of a normal player.  The input for our agent will consist of the block type information of the map.  The output will be the chosen movement of the agent. We will aim for the agent to take the shortest possible route during the challenge run. 

##AI / ML Algorithms:
Reinforcement learning with neural network.

##Evaluation Plan
We will evaluate our agent based on the percent of the map completed by the agent and time required to complete each section of the map.  There will be checkpoints along that map that indicates the agent’s progress in clearing different sets of challenges.  The agent will receive negative rewards when hitting an obstacle/enemy, dying, or receiving a slower time at each checkpoint.  The agent will receive positive reward when successfully evading obstacle/enemy, and having a faster time at each checkpoint. 

We will run through the challenge map several times using the Minecraft GUI to establish an average percent of the map completed (completion percentage) and time needed to complete that portion of the map (completion time).  The completion percentage and completion time will both serve as baselines for the level of success an agent is expected to have during this run. After implementing our algorithm, we will evaluate the average completion percentage and completion time over 10 runs to measure its effectiveness. We will verify our project works by ensuring that our algorithm is able to decrease the average completion time by 10% without decreasing the average completion percentage.  We will visualize our agent’s progress through a choice vs. time-taken graph, which would depict if the agent is successfully choosing the shortest path to the goal.  Our moonshot case will be to have the agent complete the map 25% faster than the baseline time on average.

##Appointment with the Instructor
April 22nd 9:45am

