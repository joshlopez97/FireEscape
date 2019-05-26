---
layout: default
title:  Home
---

#### Source Code
[https://github.com/joshlopez97/FireEscape](https://github.com/joshlopez97/FireEscape)

#### Summary of *Fire Escape!* Game
*Fire Escape!* is a Malmo mini game in which an agent must travel from a starting destination to an end destination without falling off of the map or dying from fire damage. The map is can elevated surface of blocks which includes a start block, an end block, ordinary path blocks, and blocks that are lit on fire. The game is won if the agent is able to travel from the start block to the end block without dying. 

<img src="https://raw.githubusercontent.com/joshlopez97/FireEscape/master/docs/img/game_won2.gif" alt="game_won2.gif" style="height: 270px;">
  
Before the start of each game, the agent has 10.0 units of health (represented by hearts). The agent will incur 0.5 units of damage per half-second if they are standing directly on fire. After standing on fire, the agent will be set on fire and will incur 0.5 units of damage per second for 8 seconds. Thus, it is possible for the agent to win the game if they are only in the fire for a brief moment.
  
<img src="https://raw.githubusercontent.com/joshlopez97/FireEscape/master/docs/img/game_won1.gif" alt="game_won1.gif" style="height: 270px;">
  
However, if the agent incurs too much damage, they will die and fail the game. Likewise, if the agent falls off the surface of blocks, they will die and fail the game. 
  
<img style="display:inline-block" src="https://raw.githubusercontent.com/joshlopez97/FireEscape/master/docs/img/game_loss1.gif" alt="game_loss1.gif" style="height: 270px;">
<img style="display:inline-block" src="https://raw.githubusercontent.com/joshlopez97/FireEscape/master/docs/img/game_loss2.gif" alt="game_loss2.gif" style="height: 270px;">

#### Relevant Links
- [Fire Damage in Minecraft](https://minecraft.gamepedia.com/Damage#Fire)
- [Project Malmo](https://github.com/microsoft/malmo)

#### Reports
- [Proposal](proposal.html)
- [Status Report](status.html)
- [Final](final.html)
