# Chess elo prediction project

Le Wagon - Data Science and AI bootcamp - batch 1812

Project contributors: Alain Lim, Jules Saint-André, Nicolas Juul, François Didden

This project was the final delivery of a 9-week full-time intensive bootcamp focused on learning Data Science with Python.

-----------------

## Predict elo scores of 2 players in a given chess game

The objective of the project was to be able to predict the elo scores of white and black players in a unique chess game.

Not all games are played at the same level and players need to know their instant level played to challenge themselves. 

Tools already exist but unfortunately they are related to a specific platform or website such as chess.com's evaluation after each game.

Here the goal was to evaluate any type of game with the same methodology and be able to compare and monitor progress over time and cross-platforms.

### Dataset

To train our models we extracted a dataset of games and elo score evaluations of games played from chess.com.

In total it represents a sample of 1.4M games.

Each game data is a string format data containing: white and black elo scores (our target), pgn (sequence of moves played), type of game, duration between moves, result of the game.

### Preprocessing approach

The preprocessing is divided in 2 parts:
* extrating a clean pgn sequence from the initial data
* transforming the pgn sequence into chess board matrices (8x8) - one board per move

Following this transformation the data is a list of matrices representing the positions the white and black chess pieces.

### Models used

### Challenges

### Conclusion



