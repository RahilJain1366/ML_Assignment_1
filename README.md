# ML_Assignment_1
In this project we will consider neural networks: first a Feedforward Neural Network (FFNN) and second a Recurrent Neural Network (RNN), for performing a 5-class Sentiment Analysis task. The objective is to predict the sentiment score (ranging from 1 to 5) of the review text.

**For running the FNN:** ``python ffnn.py --hidden_dim 16 --epochs 50 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data Data_Embedding/test.json ``

**For running the RNN:** ``python rnn.py --hidden_dim 32 --epochs 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json``

