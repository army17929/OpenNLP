import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os 
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer,set_seed,Trainer,BitsAndBytesConfig
from transformers import Trainer,TrainingArguments
import huggingface_hub
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load the model from HF
def load_model(gpu_id:int,checkpoint,bnb_config):
    torch.cuda.set_device(gpu_id)
    max_memory=f'{240000}MB'
    num_GPUs=1
    model=AutoModel.from_pretrained(checkpoint,
                        quantization_config=bnb_config,
                        max_memory={i:max_memory for i in range(num_GPUs)})
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token=tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

# Let us define a bnb configuration for quantization.
def create_bnb_config():
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='fp4',
        bnb_4bit_compute_dtype=torch.bfloat16)
    return bnb_config


# Let us do some modification for the classification problem.
class CustomClassificationModel(torch.nn.Module):
    def __init__(self,checkpoint,num_class):
        super(CustomClassificationModel,self).__init__()
        self.model=checkpoint
        self.fc1=torch.nn.Linear(self.model.config.hidden_size,16)
        self.fc2=torch.nn.Linear(16,num_class)
        self.dropout=torch.nn.Dropout(0.1)

    def forward(self,input,mask):
        # This is for the forward propagation.
        outputs=self.model(input,mask) # output from the pretrained model 
        last_hidden_state=outputs.last_hidden_state
        pooled_output=torch.mean(last_hidden_state,dim=1)
        pooled_output=pooled_output.to(torch.float32)
        x=F.relu(self.fc1(pooled_output))
        output=F.softmax(self.fc2(x),dim=1)
        x=self.dropout(x)
        return output

# Let us do some modification for the classification problem.
class GPTCustomClassificationModel(torch.nn.Module):
    def __init__(self,checkpoint,num_class):
        super(GPTCustomClassificationModel,self).__init__()
        self.model=checkpoint
        self.fc1=torch.nn.Linear(self.model.config.hidden_size,16)
        self.fc2=torch.nn.Linear(16,num_class)
        self.dropout=torch.nn.Dropout(0.1)

    def forward(self,input,mask):
        # This is for the forward propagation.
        outputs=self.model(input_ids=input,attention_mask=mask) # output from the pretrained model 
        last_hidden_state=outputs.last_hidden_state
        pooled_output=torch.mean(last_hidden_state,dim=1)
        pooled_output=pooled_output.to(torch.float32)
        x=F.relu(self.fc1(pooled_output))
        output=F.softmax(self.fc2(x),dim=1)
        x=self.dropout(x)
        return output

# Let us define some functions for PEFT.
def find_all_linear_names(model):
    # This function will return the list of modules that lora will be applied.
    cls=torch.nn.Linear
    lora_module_names=set() # Create an empty set.
    for name,module in model.named_modules():
        if isinstance(module,cls):
            names=name.split('.')
            lora_module_names.add(names[0] if len(names)==1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Based on the module list above, we can create a configuration
def create_peft_config(modules):
    peft_config=LoraConfig(r=512,lora_alpha=32,
                           target_modules=modules,lora_dropout=0.1,
                           bias='none')
    return peft_config

# We can check the percentage of the trainable parameters
def print_trainable_parameters(model,use_4bit=False):
    trainable_params=0
    all_param=0
    for _,param in model.named_parameters():
        num_params=param.numel()
        if num_params==0 and hasattr(param,"ds_numel"):
            num_params=param.ds_numel
        all_param+=num_params
        if param.requires_grad:
            trainable_params+=num_params
    if use_4bit:
        trainable_params/=2
    print(
    f"all params:{all_param:,d} || trainable params:{trainable_params:,d}"
    f"\n trainable % = {100*trainable_params/all_param}")

# Define a function for the model training 
def train_preparation(model):

    # Prepare the model for the kbit training
    model=prepare_model_for_kbit_training(model)

    # Find out the modules and layers that lora can be applied.
    target_modules=find_all_linear_names(model)
    peft_config=create_peft_config(target_modules) # Create configuration
    model=get_peft_model(model,peft_config=peft_config) # Model modification done.
    print('MODEL SUMMARY \n',model)
    print_trainable_parameters(model)
    return model

