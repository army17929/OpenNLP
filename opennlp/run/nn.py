import pandas as pd 
import numpy as np
from sklearn import neural_network 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Conv1D,GlobalMaxPool1D
from keras.utils import to_categorical
from opennlp.run.ml import ClassicalML
from opennlp.trainer.trainer import metrics_generator, binary_metrics_generator, PlotTraining
import matplotlib.pyplot as plt
import time

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
    def __init__(self,
                 user_split:bool,
                 input_col:str,output_col:str,
                 num_epochs:int,
                 num_nodes:int,num_layers:int,bs:int,
                 lineterminator=None,
                 data_path=None,
                 train_filepath=None,
                 test_filepath=None,
                 input_length=64,
                 encoding='utf-8'):
        self.user_split=user_split
        if self.user_split==False:
            self.df=pd.read_csv(data_path,
                                lineterminator=lineterminator,
                                encoding=encoding,
                                encoding_errors='ignore')
            self.X=self.df[input_col] # Raw text
            self.y=self.df[output_col]
            self.num_class=len(self.y.unique())
        if self.user_split:
            self.df_train=pd.read_csv(train_filepath,
                                      lineterminator=lineterminator,
                                      encoding=encoding,
                                      encoding_errors='ignore')
            self.df_test=pd.read_csv(test_filepath,
                                      lineterminator=lineterminator,
                                      encoding=encoding,
                                      encoding_errors='ignore')
            self.X_train=self.df_train[input_col]
            self.X_test=self.df_test[input_col]
            self.y_train=self.df_train[output_col]
            self.y_test=self.df_test[output_col]
            print(f"X_train {self.X_train}")
            print(f"X_test {self.X_test}")
            print(f"y_train {self.y_train}")
            print(f"y_test {self.y_test}")
            self.num_class=len(self.y_train.unique())
        self.total_epochs=num_epochs
        self.batch_size=bs
        self.tokenizer=Tokenizer()
        self.num_layers=num_layers
        self.num_nodes=num_nodes
        self.model=Sequential()
        self.input_length=input_length

    def tokenize(self,X,y):
        self.tokenizer.fit_on_texts(X)
        X_sequence=self.tokenizer.texts_to_sequences(X)
        X_padded=pad_sequences(X_sequence,maxlen=self.input_length)
        y_encoded,y_classes=pd.factorize(y)
        y_categorical=to_categorical(y_encoded)
        return X_padded,y_categorical
    
    def build_model(self,input_length):
        """
        LSTM Generation function 

        :param input_length: (int) maxumum length of input data. This will be input dimension of the embedding layer.
        :return model: (object) LSTM 
        """
        self.model.add(Embedding(input_dim=self.input_length,
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
        if self.user_split==False:
            X,y=self.tokenize(self.X,self.y)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
            model=self.build_model(input_length=self.input_length)
        if self.user_split:
            X_train,y_train=self.tokenize(self.X_train,self.y_train)
            self.X_test=self.X_test.astype(str)
            X_test,y_test=self.tokenize(self.X_test,self.y_test)
            print(f"y_test {y_test}")
            print(f"X_test {X_test}")
            model=self.build_model(input_length=self.input_length)

        # Compile the model.
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        start=time.time()
        history=model.fit(X_train,y_train,
                          epochs=self.total_epochs,
                          batch_size=self.batch_size,
                          validation_split=0.1)
        print(f"HISTORY : {history}")
        end=time.time()
        runtime=end-start
        print(f"RUNTIME {runtime:.2f} sec")
        PlotTraining(history=history,
                     savedir=f"/LSTM_{self.num_layers}layers_{self.num_nodes}nodes",
                    model_name="LSTM")
        logit=model.predict(X_test)
        y_pred=np.argmax(logit,axis=1)
        y_true=np.argmax(y_test,axis=1)
        print("prediction",y_pred)
        print("true labels",y_true)
        report=classification_report(y_pred=y_pred,y_true=y_true)
        print(report)
        if self.num_class==2:
            binary_metrics_generator(y_true=y_true,y_pred=y_pred,
                          save_dir=f"/LSTM_{self.num_layers}layers_{self.num_nodes}nodes",
                          model_name="LSTM",runtime=runtime)    
        else:
            metrics_generator(y_true=y_true,y_pred=y_pred,
                          save_dir=f"/LSTM_{self.num_layers}layers_{self.num_nodes}nodes",
                          model_name="LSTM",runtime=runtime)

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
    def __init__(self,
                 user_split:bool,
                 input_col:str, output_col:str,bs:int,lr:float,
                 hidden_layer_sizes:set,max_iter:int,
                 data_path=None,
                 lineterminator=None,
                 train_filepath=None,
                 test_filepath=None,
                 encoding='utf-8'):
        self.user_split=user_split
        self.input_col=input_col
        self.output_col=output_col
        if self.user_split==False:
            self.df=pd.read_csv(data_path,
                                lineterminator=lineterminator,
                                encoding=encoding,
                                encoding_errors='ignore')
            self.X=self.df[input_col] # Raw text
            self.y=self.df[output_col]       
            self.num_class=len(self.df[self.output_col].unique())  
        if self.user_split:
            self.df_train=pd.read_csv(train_filepath,
                                      lineterminator=lineterminator,
                                      encoding=encoding,
                                      encoding_errors='ignore')
            self.df_test=pd.read_csv(test_filepath,
                                      lineterminator=lineterminator,
                                      encoding=encoding,
                                      encoding_errors='ignore')
            self.X_train=self.df_train[input_col]
            self.X_test=self.df_train[output_col]
            self.y_train=self.df_test[input_col]
            self.y_test=self.df_test[output_col]
            self.num_class=len(self.df_train[self.output_col].unique())   
        self.data_path=data_path
        self.train_filepath=train_filepath
        self.test_filepath=test_filepath
        self.hidden_layer_sizes=hidden_layer_sizes
        self.num_layers=len(hidden_layer_sizes)
        self.batch_size=bs
        self.learning_rate=lr
        self.model=neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                max_iter=max_iter)
    
    def run_MLP(self):
        if self.user_split==False:
            ml=ClassicalML(data_path=self.data_path,
                        input_col=self.input_col,
                        output_col=self.output_col,
                        seed=42)
            X=ml.vectorize_data(self.input_col)
            y=self.df[self.output_col]
            X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                test_size=0.2)
        if self.user_split==True:
            ml_train=ClassicalML(data_path=self.train_filepath,
                    input_col=self.input_col,
                    output_col=self.output_col,
                    seed=42)
            ml_test=ClassicalML(data_path=self.test_filepath,
                                input_col=self.input_col,
                                output_col=self.output_col,
                                seed=42)
            X_train=ml_train.vectorize_data(self.input_col)
            X_test=ml_test.vectorize_data(self.input_col)
            y_train=ml_train.vectorize_data(self.output_col)
            y_test=ml_test.vectorize_data(self.output_col)
        model=self.model
        start=time.time()
        model.fit(X_train,y_train)
        end=time.time()
        runtime=end-start
        print(f"RUNTIME : {runtime}")
        y_pred=model.predict(X_test)
        report=classification_report(y_pred=y_pred,y_true=y_test)
        if self.num_class==2:
            binary_metrics_generator(y_true=y_test,
                          y_pred=y_pred,
                          save_dir=f'/MLP_{len(self.hidden_layer_sizes)}layers',
                          model_name='MLP',
                          runtime=runtime)
        else:
            metrics_generator(y_true=y_test,
                          y_pred=y_pred,
                          save_dir=f'/MLP_{len(self.hidden_layer_sizes)}layers',
                          model_name='MLP',
                          runtime=runtime)

class CNN():
    def __init__(self,
                 user_split:bool,
                 input_col:str,output_col:str,
                 num_epochs:int,num_nodes:int,num_layers:int,
                 num_conv:int,filter:int,kernel:int,
                 bs:int,lineterminator=None,
                 input_length=32,
                 data_path=None,
                 train_filepath=None,
                 test_filepath=None,
                 encoding='utf-8'):
        self.user_split=user_split
        if self.user_split==False:
            self.df=pd.read_csv(data_path,
                                encoding=encoding,
                                lineterminator=lineterminator,
                                encoding_errors='ignore')
            self.X=self.df[input_col]
            self.y=self.df[output_col]
            self.num_class=len(self.y.unique())
        if self.user_split==True:
            self.df_train=pd.read_csv(train_filepath,
                                      encoding=encoding,
                                      lineterminator=lineterminator,
                                      encoding_errors='ignore')
            self.X_train=self.df_train[input_col]
            self.y_train=self.df_train[output_col]
            self.df_test=pd.read_csv(test_filepath,
                                     encoding=encoding,
                                     lineterminator=lineterminator,
                                     encoding_errors='ignore')
            self.X_test=self.df_test[input_col]
            self.y_test=self.df_test[output_col]
            self.num_class=len(self.y_train.unique())
        self.total_epochs=num_epochs
        self.num_nodes=num_nodes
        self.num_layers=num_layers
        self.bs=bs
        self.kernel=kernel
        self.filter=filter
        self.num_conv=num_conv
        self.input_length=input_length
        self.tokenizer=Tokenizer()
        self.model=Sequential()
    
    def tokenize(self,X,y):
        self.tokenizer.fit_on_texts(X)
        X_sequence=self.tokenizer.texts_to_sequences(X)
        X_padded=pad_sequences(X_sequence,maxlen=self.input_length)
        y_encoded,y_classes=pd.factorize(y)
        y_categorical=to_categorical(y_encoded)
        return X_padded,y_categorical
    
    def build_model(self):
        self.model.add(Embedding(input_dim=self.input_length,
                                 output_dim=self.num_nodes,
                                 input_length=self.input_length))
        i=1
        # Convloution layers
        while(i<=self.num_conv):
            self.model.add(Conv1D(filters=self.filter,
                                  kernel_size=self.kernel,
                                  activation='relu'))
            i+=1
        # Max pooling layer
        self.model.add(GlobalMaxPool1D())
        # Hidden layers
        j=1
        while(j<=self.num_layers):
            self.model.add(Dense(self.num_layers,
                                 activation='relu'))
            j+=1
        self.model.add(Dense(self.num_class,
                             activation='softmax'))
        print(self.model.summary)
        return self.model
    
    def run_CNN(self):
        if self.user_split==False:
            X,y=self.tokenize(self.X,self.y)
            X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                           test_size=0.2,
                                                           random_state=42)
        if self.user_split:
            X_train,y_train=self.tokenize(self.X_train,self.y_train)
            self.X_test=self.X_test.astype(str)
            X_test,y_test=self.tokenize(self.X_test,self.y_test)
        model=self.build_model()
        # Complile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # Fit the model
        start=time.time()
        history=model.fit(X_train,y_train,
                          epochs=self.total_epochs,
                          batch_size=self.bs,
                          validation_split=0.1)
        end=time.time()
        runtime=end-start
        print(f"RUNTIME {runtime:.2f} sec")
        PlotTraining(history=history,
                     savedir=f"/CNN_{self.num_layers}layers_{self.num_nodes}nodes",
                    model_name="CNN")
        logit=model.predict(X_test)
        y_pred=np.argmax(logit,axis=1)
        y_true=np.argmax(y_test,axis=1)
        report=classification_report(y_pred=y_pred,y_true=y_true)
        print(report)
        if self.num_class==2:
            binary_metrics_generator(y_true=y_true,y_pred=y_pred,
                          save_dir=f"/CNN_{self.num_layers}layers_{self.num_nodes}nodes",
                          model_name="CNN",runtime=runtime)    
        else:
            metrics_generator(y_true=y_true,y_pred=y_pred,
                          save_dir=f"/CNN_{self.num_layers}layers_{self.num_nodes}nodes",
                          model_name="CNN",runtime=runtime)

