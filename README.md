# Catan Bot
A side project to use AI to give suggestions for initial placements in Settlers of Catan. This project was used to fulfill the project requirement for 16-811 (Robomath) at CMU.

## How to run the board placement script:
1. Clone this repo somewhere.
2. pip install it (`pip3 install .` from the base dir).
3. Move to the ui directory (`cd catanbot/ui/`).
4. Run the board placement script (`python3 board_placement_ui_with_vf.py (--nthreads <# threads> --c <exploration factor> --max_time <max time to run>) --net_path <path to the trained neural network>`).
### Args for `board_placement_ui_continuous.py`
1. `--nthreads`: The number of processes to run MCTS on
2. `--c`: The exploration factor for MCTS
3. `--max_time`: The maximum amount of time (in seconds) to let MCTS run. The MCTS is also interruptable with ^C. 
4. `--nsamples`: The number of games to play at each rollout
5. `--net_path`: The path to the neural network for value estimation (A trained network is provided in `catanbot/ui/trained_qf.cpt`)

Tiles can be edited by clicking on them (they should be highlighted) and inputting a string of the following format: `<dice value><resource type>`. Resource types are one of the following: O = ore, W = wheat, S = sheep, L = wood, B = brick, D = desert. Clicking on a port and inputting one of these charaters will change the port to be for whatever resource you selected (A = 3:1 port). Clicking on a settlement spot or road spot will place a settlement/road there. Clicking again will cycle through each player. You can run MCTS for placements by clicking the 'Run MCTS' button. This will prompt the user for input in the terminal and will them run for the time specified. The top 5 choices are placed on the UI as arrows from the settlement spot in the direction of the road placement. You can also display the production if each settlement spot (sum of the pips of adjacent hexes) with the 'Toggle Production Values' button.

## Status:
1. Catan simulator: Done.
2. Catan agents: Moves according to a heuristic. Will place cities and settlements whenever possible (highest-producing spots first). Will build roads towards available settlement spots if no other settlement spots already available.
3. Board placement AI: Essentially done. Uses MCTS to search for good settlement spots for any player in the initial placement order.
