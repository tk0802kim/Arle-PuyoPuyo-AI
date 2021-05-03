# Arle - PuyoPuyo AI using PuyoPuyo 2 rules

It generally obeys PuyoPuyo2 rules except two notable exceptions:
    1. Columns can't be blocked off. For example, if column 3 is filled to the top, the game won't let you place anything in columns>=4, but here it is allowed for simplicity's sake.
    2. No bonus for pressing downkey


puyo class in Puyo.py is the actual instance of the game. It contains all the game state related attributes, as well as functions to evolve the gamestate.

In particular:
    puyo.place(): places the puyo at the top
    puyo.chain(): runs through the sequence of events including chaining and calculating the score
    puyo.newgame(): start a blank instance of a new game
    
    
DQL.py contains classes and functions to be used in deep Q learning.(https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

In particular:
    memory class: contains state, action taken, reward observed, and the resulting state. This will be collected in an array called memeory_lane in the main function



PuyoPuyo is a player vs player tetris-like game. When 4 or more blocks(Puyos) of the same color are connected, they explode, generating score and creating space for Puyos to drop down. Scores are then translated into garbage puyo's which are sent to the opponent's gamespace to obstruct. Here is a gameplay video:

[![](http://img.youtube.com/vi/ZbWQ36NWSNQ/0.jpg)](http://www.youtube.com/watch?v=ZbWQ36NWSNQ "")
