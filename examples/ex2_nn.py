"""
            EX 2. Running Neural Network models using OpenNLP

This is an tutorial for Neural Network for NLP.
OpenNLP includes LSTM and MLP. (CNN will be included soon)
1. LSTM 
2. MLP
"""

# Import dependencies 
from opennlp.run.nn import _LSTM, MLP

# Create an instance from LSTM class
lstm=_LSTM(data_path='./data/sample_sentiment.csv',
           input_col='tweets',output_col='labels',
           num_epochs=20,num_nodes=30,num_layers=10,bs=32)
# Run LSTM on sentiment dataset
lstm.run_LSTM()

mlp=MLP(data_path='./data/sample_sentiment.csv',
        input_col='tweets',output_col='labels',
        bs=32,
        lr=1e-5,
        hidden_layer_sizes=(20,40,60), # 3 Layers, Each layer has 20,40,60 nodes
        max_iter=1000) # Note that MLP does not have epochs, max iteration number will control the training

# Run MLP
mlp.run_MLP()

"""
Results of those models will be saved at './Results/<model name>_<parameters>'
For example, if you run Random forest model with 200 estimators,
Results will be saved in ./Results/LSTM_10layers_30nodes
Results include runtime,confusion matrix,clcassification report and training and validation loss.
"""

