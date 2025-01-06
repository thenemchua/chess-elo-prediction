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

Each game data is a string format data including: 
* white and black elo scores (our target)
* pgn (sequence of moves played)
* game type
* duration between moves
* game result.

### Preprocessing approach

The preprocessing is divided in 2 parts:
* extrating a clean pgn sequence from the raw data
* transforming the pgn sequence into chess board matrices (8x8) - each matrix representing a board state after a move.

Following this transformation the data is a list of matrices representing the positions of the white and black chess pieces.

### Models used

Our baseline model is an LSTM reading the pgn sequences as string format.

Then after transforming the pgn into a list of n matrices - n being the number of moves played in the game - we used a combination of CNN and LSTM.

The model first read the matrices using layers of CNN extracting the spatial information, then the duration between moves is added to the output of the CNN and goes through a few layers of LSTM to model the temporal dynamics of the game.

### Challenges

A few challenges were faced in the projetc:
* size of the dataset - our local machines were not able to handle the full dataset. Virtual Machines (VMs) on Google Cloud Platform (GCP) were used for processing
* duration of model training - each epoch equired several hours.

### Conclusion

The model succeeded in predicting diferent white and black elo scores using a single pgn sequence.

The Mean Absolute Error (MAE) was reduced using the CNN/LSTM model and the performance exceeded the initial project goals.

