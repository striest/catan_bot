# Catan Bot
A side project to use AI to give suggestions for initial placements in Settlers of Catan. 

## How to run the board placement script:
1. Clone this repo somewhere.
2. pip install it (`pip3 install .` from the base dir).
3. Move to the ui directory (`cd catanbot/ui/`).
4. Run the board placement script (`python3 board_placement_ui_continuous.py`).
### Args for `board_placement_ui_continuous.py`
1. `--nthreads`: The number of processes to run MCTS on
2. `--c`: The exploration factor for MCTS
3. `--max_time`: The maximum amount of time (in seconds) to let MCTS run. The MCTS is also interruptable with ^C. 

You can cycle through resource tiles/values and ports by clicking on them. Clicking on a settlement spot or road spot will place a settlement/road there. Clicking again will cycle through each player. You can run MCTS for placements by clicking the 'Run MCTS' button. This will prompt the user for input in the terminal and will them run for the time specified. The top 5 choices are placed on the UI as arrows from the settlement spot in the direction of the road placement. You can also display the production (sum of the pips of adjacent hexes) with the 'Toggle Production Values' button.

## Status:
1. Catan simulator: Mostly done - need to add dev cards, longest road/largest army, robber.
2. Catan agents: Moves according to a heuristic. Will place cities and settlements whenever possible (highest-producing spots first). Will build roads towards available settlement spots if no other settlement spots already available.
3. Board placement AI: Essentially done. Uses MCTS to search for good settlement spots for any player in the initial placement order.

## TODO:
1. Finish the simulator.
2. Re-design the board placement script to run MCTS in the background and reuse the search tree for multiple queries.
