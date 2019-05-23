# Malmo Fire Escape
### How the game is played
*Fire Escape* is a Malmo mini game in which an agent must travel from a starting destination to an end destination without falling off of the map or dying from fire damage. The map is can elevated surface of blocks which includes a start block, an end block, ordinary path blocks, and blocks that are lit on fire. If an agent walks on a block that is lit on fire, the agent is lit on fire for a short period of time. During this period of time, the user will incur damage. If the agent incurs too much damage, they will die and fail the game. Likewise, if the agent falls off the surface of blocks, they will die and fail the game. The game is won if the agent is able to travel from the start block to the end block without dying.
### Evaluating Performance
Although the primary concern of the agent is to reach the end block without dying, we are also interested in tracking the number of moves needed to finish the game. A "move" counts as moving one block north, south, east, or west. We will evaluate performance based on: (1) whether or not the agent reached the end block without dying and (2) the number of moves needed to finish the game.

### Website
https://joshlopez97.github.io/Malmo-Fire-Escape/
