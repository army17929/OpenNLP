from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn 
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import os

# checkpoint='meta-llama/Llama-2-7b-hf'

def load_preprocessed_nuclear_data():
    file='Tweets_7tools_preprocessed_dropbox.csv'
    df=pd.read_csv(file)
    df=df[['tweets','FinalScore']]
    df['FinalScore']=df['FinalScore'].replace('Positive',1)
    df['FinalScore']=df['FinalScore'].replace('Negative',2)
    df['FinalScore']=df['FinalScore'].replace('Neutral',0)
    print('Nuclear data is loaded. This data contains tweets and label(int)')
    return df

def data_analyzer(df,output_col:str,savedir:str,filename:str):
    count=df[output_col].value_counts()
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

# Conversion from text to tensor.
def prepare_dataset(X,y,checkpoint,max_length=128):

    # Import the tokenizer for the argument 
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token=tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # This is for BERT

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1)
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

# Conversion from text to tensor.
def prepare_dataset_BERT(X,y,checkpoint,max_length=128):

    # Import the tokenizer for the argument 
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # This is for BERT

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1)
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
