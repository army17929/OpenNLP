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
from Classical_ML_Engine import vectorize_data
from Trainer import metrics_generator
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from Trainer import PlotTraining

def set_worker(gpu_index:int):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set device
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

class _LSTM():
    def __init__(self,df,input_col:str,label_col:str,
                 num_class:int,num_epochs:int,
                 num_nodes:int,num_layers:int,bs:int):
        self.X=df[input_col] # Raw text
        self.y=df[label_col]
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
    def __init__(self,df,input_col:str, 
                 label_col:str,bs:int,lr:float,
                 hidden_layer_sizes:set,max_iter:int):
        self.df=df
        self.input_col=input_col
        self.label_col=label_col
        self.hidden_layer_sizes=hidden_layer_sizes
        self.num_layers=len(hidden_layer_sizes)
        self.batch_size=bs
        self.learning_rate=lr
        self.model=neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                max_iter=max_iter)
    
    def run_MLP(self):
        X=vectorize_data(self.df,self.input_col)
        y=self.df[self.label_col]
        X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2)
        model=self.model
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        metrics_generator(y_true=y_test,
                          y_pred=y_pred,
                          save_dir=f'/MLP_{len(self.hidden_layer_sizes)}layers',
                          model_name='MLP')



