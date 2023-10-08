import pandas as pd 
import numpy as np
from sklearn import neural_network 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
from keras.utils import to_categorical
import os 
from classical_ml import ClassicalML
from Trainer import metrics_generator
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from Trainer import PlotTraining

class _LSTM():
    """
    :param df: (DataFrame) data for training
    :param input_col: (str) name of the column that contains input data
    :param output_col: (str) name of the column that contains output data
    :param num_classes: (int) number of classes in the classification problem
    :param num_epochs: (int) total number of epochs
    :param num_nodes: (int) number of nodes in the layers
    :param num_layers: (int) number of hidden layers in LSTM
    :param bs: (int) batch size for training
    """
    def __init__(self,df,input_col:str,output_col:str,
                 num_class:int,num_epochs:int,
                 num_nodes:int,num_layers:int,bs:int):
        self.X=df[input_col] # Raw text
        self.y=df[output_col]
        self.total_epochs=num_epochs
        self.batch_size=bs
        self.tokenizer=Tokenizer()
        self.num_class=num_class # Classification
        self.num_layers=num_layers
        self.num_nodes=num_nodes
        self.model=Sequential()

    def tokenize(self,X,y):
        self.tokenizer.fit_on_texts(X)
        X_sequence=self.tokenizer.texts_to_sequences(X)
        X_padded=pad_sequences(X_sequence)
        y_encoded,y_classes=pd.factorize(y)
        y_categorical=to_categorical(y_encoded)
        return X_padded,y_categorical
    
    def build_model(self,input_length):
        """
        LSTM Generation function 

        :param input_length: (int) maxumum length of input data. This will be input dimension of the embedding layer.
        :return model: (object) LSTM 
        """
        self.model.add(Embedding(input_dim=len(self.tokenizer.word_index)+1,
                output_dim=self.num_nodes,
                input_length=input_length))
        for i in range (self.num_layers) :
            if i<self.num_layers-1:
                self.model.add(LSTM(self.num_nodes,return_sequences=True))
            else:
                self.model.add(LSTM(self.num_nodes,return_sequences=False))
        self.model.add(Dense(self.num_class,activation='softmax'))
        print(self.model.summary())
        return self.model
    
    def run_LSTM(self):
        #"""
        #LSTM training function 
        #"""
        X,y=self.tokenize(self.X,self.y)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
        model=self.build_model(input_length=X.shape[1])
        
        # Compile the model.
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        start=time.time()
        history=model.fit(X_train,y_train,epochs=self.total_epochs,batch_size=self.batch_size,validation_split=0.1)
        end=time.time()
        print(f"RUNTIME {end-start:.2f} sec")
        PlotTraining(history=history,
                     savedir=f"/LSTM_{self.num_layers}layers_{self.num_nodes}nodes",
                    model_name="LSTM")
        logit=model.predict(X_test)
        y_pred=np.argmax(logit,axis=1)
        y_true=np.argmax(y_test,axis=1)
        metrics_generator(y_true=y_true,y_pred=y_pred,
                          save_dir=f"/LSTM_{self.num_layers}layers_{self.num_nodes}nodes",
                          model_name="LSTM")

class MLP():
    """
    Multi Layers Perceptron module 

    :param df: (DataFrame) data for training 
    :param input_col: (str) name of the column that contains input data
    :param output_col: (str) name of the column that contains output data
    :param bs: (int) batch size 
    :param lr: (float) learning rate
    :param hidden_layer_sizes: (set) set of number of nodes in the layers. Example: ``(20,30,50)`` for model with 3 hidden layers
    :param max_iter: (int) maximum number of iteration
    """
    def __init__(self,df,input_col:str, 
                 output_col:str,bs:int,lr:float,
                 hidden_layer_sizes:set,max_iter:int):
        self.df=df
        self.input_col=input_col
        self.output_col=output_col
        self.hidden_layer_sizes=hidden_layer_sizes
        self.num_layers=len(hidden_layer_sizes)
        self.batch_size=bs
        self.learning_rate=lr
        self.model=neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                max_iter=max_iter)
    
    def run_MLP(self):
        ml=ClassicalML(df=self.df,input_col=self.input_col,output_col=self.output_col,seed=42)
        X=ml.vectorize_data(self.df,self.input_col)
        y=self.df[self.output_col]
        X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2)
        model=self.model
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        metrics_generator(y_true=y_test,
                          y_pred=y_pred,
                          save_dir=f'/MLP_{len(self.hidden_layer_sizes)}layers',
                          model_name='MLP')



