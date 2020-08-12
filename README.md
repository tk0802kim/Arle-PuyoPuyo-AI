# PuyoPuyo
PuyoPuyo AI using PuyoPuyo2 rules
It obeys every rule in Puyo2 except two notable exceptions:
    1. Columns can't be blocked off. For example, if column 3 is filled to the top, the game won't let you place anything in columns>=4, but here it is allowed for simplicities sake.
    2. no bonus for pressing downkey



puyo class in Puyo.py is the actual instance of the game. It contains all the game state related attributes, as well as functions to evolve the gamestate.

In particular:
    puyo.place(): places the puyo at the top
    puyo.chain(): runs through the sequence of events including chaining and calculating the score
    puyo.newgame(): start a blank instance of a new game
    
    
DQL.py contains classes and functions to be used in deep Q learning.(https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

In particular:
    memory class: contains state, action taken, reward observed, and the resulting state. This will be collected in an array called memeory_lane in the main function



ToDo:
Parallelize



Done:
Make it so the initial training set is randomly generated state, not a blank state that evolves in time.
Investigate memory overflow
