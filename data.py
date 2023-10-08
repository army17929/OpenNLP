from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn 
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import os

class data_processor():
    """
    Data Processor module

    :param path: (str) path where data exists
    :param input_col: (str) name of the column that contains input data
    :param output_col: (str) name of the column that contains output data
    :param encoding: (str) encoding type. Example: ``cp1252``
    """
    def __init__(self,path:str,input_col:str,output_col:str,encoding='utf-8'):
        self.path=path # Path of the data
        self.df=pd.read_csv(path,encoding=encoding)
        self.df=self.df[[input_col,output_col]]
        self.input_col=input_col
        self.output_col=output_col
        self.X=self.df[input_col]
        self.y=self.df[output_col]

    def label_converter(self):
        #"""
        #This function will convert sentiment labels into integers. Neutral=0,Positive=1,Negative=2 
        #"""
        self.df[self.output_col]=self.df[self.output_col].replace('Neutral',0)
        self.df[self.output_col]=self.df[self.output_col].replace('Positive',1)
        self.df[self.output_col]=self.df[self.output_col].replace('Negative',2)
        return self.df

    def data_analyzer(self,output_col:str,savedir:str,filename:str):
        #"""
        #Data analyzer 
        #This function will analyze the distribution of the data. 
        #Analyzed bar chart and pie chart will be saved in ``savedir``
        #"""
        count=self.df[output_col].value_counts()
        print(count)
        plt.clf()
        plt.figure(figsize=(12,10))
        plt.subplot(1,2,1)
        plt.title('Class Distribution - Bar')
        plt.bar(height=count,x=["Negative","Positive","Neutral"])
        plt.subplot(1,2,2)
        plt.title('Class Distribution - Pie')
        plt.pie(x=count,labels=["Negative","Positive","Neutral"],autopct='%1.1f%%')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(savedir+'/'+filename)

    def prepare_dataset(self,df,checkpoint,max_length=128,
                        test_size=0.2,val_size=0.1,seed=42):
        #"""
        #Dataset preparation function 
        #This function will convert raw data into tensor dataset.
        #Input: 
        #    df : DataFrame type object
        #    checkpoint : Tokenizer type you want to use for preprocessing. Example: ``gpt2``
        #    max_length : maximum length of input text. Default=128
        #    test_size : portion of test data
        #    val_size : portion of validation data
        #    seed : random seed for train and test split
        #Return: 
        #    - Tuple[TensorDataset,TensorDataset,TensorDataset]
        #"""

        tokenizer=AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token=tokenizer.eos_token
        
        X=df[self.input_col]
        y=df[self.output_col]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=seed)
        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=val_size,random_state=seed)
        # Prepare for input_ids
        # Make a list type object
        X_train_text=[str(text) for text in X_train]
        X_test_text=[str(text) for text in X_test]
        X_val_text=[str(text) for text in X_val]

        # Convert vocabs into tensors
        X_train_tensor=[tokenizer(str(text),return_tensors='pt',
                                max_length=max_length,truncation=True,
                                pad_to_max_length=True)['input_ids'] 
                                for text in X_train_text]
        X_test_tensor=[tokenizer(str(text),return_tensors='pt',
                            max_length=max_length,truncation=True,
                            pad_to_max_length=True)['input_ids']
                            for text in X_test_text]
        X_val_tensor=[tokenizer(str(text),return_tensors='pt',
                                max_length=max_length,truncation=True,
                                pad_to_max_length=True)['input_ids'] 
                                for text in X_val_text]

        # Convert list to tensor. 
        X_train_input=torch.squeeze(torch.stack(X_train_tensor),dim=1)
        X_test_input=torch.squeeze(torch.stack(X_test_tensor),dim=1)
        X_val_input=torch.squeeze(torch.stack(X_val_tensor),dim=1)

        # Prepare for attention masks
        X_train_mask=[tokenizer(str(text),return_tensors='pt',
                                max_length=max_length,truncation=True,
                                pad_to_max_length=True)['attention_mask']
                                for text in X_train]
        X_test_mask=[tokenizer(str(text),return_tensors='pt',
                                max_length=max_length,truncation=True,
                                pad_to_max_length=True)['attention_mask']
                                for text in X_test]
        X_val_mask=[tokenizer(str(text),return_tensors='pt',
                              max_length=max_length,truncation=True,
                              pad_to_max_length=True)['attention_mask']
                              for text in X_val]

        # Squeeze it
        X_train_mask=torch.squeeze(torch.stack(X_train_mask),dim=1)
        X_test_mask=torch.squeeze(torch.stack(X_test_mask),dim=1)
        X_val_mask=torch.squeeze(torch.stack(X_val_mask),dim=1)

        # Prepare for the label data
        y_train=torch.tensor(y_train.tolist())
        y_test=torch.tensor(y_test.tolist())
        y_val=torch.tensor(y_val.tolist())

        # Now, all the data is prepared.
        train_dataset=TensorDataset(X_train_input,X_train_mask,y_train)
        test_dataset=TensorDataset(X_test_input,X_test_mask,y_test)
        val_dataset=TensorDataset(X_val_input,X_val_mask,y_val)

        return train_dataset,test_dataset,val_dataset

    def prepare_dataset_BERT(self,df,checkpoint,max_length=128,
                             test_size=0.2,val_size=0.1,seed=42):
        #"""
        #Same function as above, but only for BERT
        #"""
        tokenizer=AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # This is for BERT
        
        X=df[self.input_col]
        y=df[self.output_col]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=seed)
        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=val_size,random_state=seed)

        # Make a list type object
        X_train_text=[str(text) for text in X_train]
        X_test_text=[str(text) for text in X_test]
        X_val_text=[str(text) for text in X_val]

        # Convert vocabs into tensors
        X_train_tensor=[tokenizer(str(text),return_tensors='pt',
                                max_length=max_length,truncation=True,
                                pad_to_max_length=True)['input_ids'] 
                                for text in X_train_text]
        X_test_tensor=[tokenizer(str(text),return_tensors='pt',
                            max_length=max_length,truncation=True,
                            pad_to_max_length=True)['input_ids']
                            for text in X_test_text]
        X_val_tensor=[tokenizer(str(text),return_tensors='pt',
                                max_length=max_length,truncation=True,
                                pad_to_max_length=True)['input_ids'] 
                                for text in X_val_text]

        # Convert list to tensor. 
        X_train_input=torch.squeeze(torch.stack(X_train_tensor),dim=1)
        X_test_input=torch.squeeze(torch.stack(X_test_tensor),dim=1)
        X_val_input=torch.squeeze(torch.stack(X_val_tensor),dim=1)

        # Prepare for attention masks
        X_train_mask=[tokenizer(str(text),return_tensors='pt',
                                max_length=max_length,truncation=True,
                                pad_to_max_length=True)['attention_mask']
                                for text in X_train]
        X_test_mask=[tokenizer(str(text),return_tensors='pt',
                                max_length=max_length,truncation=True,
                                pad_to_max_length=True)['attention_mask']
                                for text in X_test]
        X_val_mask=[tokenizer(str(text),return_tensors='pt',
                              max_length=max_length,truncation=True,
                              pad_to_max_length=True)['attention_mask']
                              for text in X_val]

        # Squeeze it
        X_train_mask=torch.squeeze(torch.stack(X_train_mask),dim=1)
        X_test_mask=torch.squeeze(torch.stack(X_test_mask),dim=1)
        X_val_mask=torch.squeeze(torch.stack(X_val_mask),dim=1)

        # Prepare for the label data
        y_train=torch.tensor(y_train.tolist())
        y_test=torch.tensor(y_test.tolist())
        y_val=torch.tensor(y_val.tolist())

        # Now, all the data is prepared.
        train_dataset=TensorDataset(X_train_input,X_train_mask,y_train)
        test_dataset=TensorDataset(X_test_input,X_test_mask,y_test)
        val_dataset=TensorDataset(X_val_input,X_val_mask,y_val)

        return train_dataset,test_dataset,val_dataset

def load_preprocessed_nuclear_data():
    #"""
    #This is a temporary helper function that can directly load most frequently used data. 
    #"""
    file='./Tweets_7tools_preprocessed_dropbox.csv'
    df=pd.read_csv(file)
    df=df[['tweets','FinalScore']]
    df['FinalScore']=df['FinalScore'].replace('Positive',1)
    df['FinalScore']=df['FinalScore'].replace('Neutral',0)
    df['FinalScore']=df['FinalScore'].replace('Negative',2)
    print('Nuclear data is loaded. This data contains tweets and label(int)')
    return df