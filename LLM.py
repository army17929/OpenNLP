from trainer import TrainerSingle,TrainerDDP,Trainer_multinode,prepare_const,ddp_setup,ddp_setup_torchrun
from model import CustomClassificationModel, peft, PEFTClassificationModel
from data import data_processor
import os 
import time
from pathlib import Path
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BERT():
    """
    Bidirectional Encoder Representations from Transformers (BERT) module 

    :param df: (DataFrame) raw data you want to use for fine tuning. Contains text(str) and labels(int).
    :param input_col: (str) name of the column that contains input text.
    :param output_col: (str) name of the column that contains output label.
    :param max_length: (int) maximum length of the input text. If the input exceeds maximum, the model will cut off.
    :param test_size: (float) portion of test data for model evalutation.
    :param val_size: (float) portion of validation data for training evaluation. 
    :param seed: (int) random seed for train and test split. 
    """

    def __init__(self,data_path:str,input_col:str,output_col:str,num_class:int,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42,encoding='utf-8'): 
        D=data_processor(path=data_path,input_col=input_col,
                            output_col=output_col,encoding=encoding)
        self.df=pd.read_csv(data_path,encoding=encoding)
        self.df=D.label_converter()
        self.checkpoint='bert-base-uncased' # model checkpoint
        self.X=self.df[input_col] 
        self.y=self.df[output_col] 
        self.num_class=num_class
        self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset_BERT(checkpoint=self.checkpoint)
        
    def run_BERT(self,epochs:int,bs:int,lr:float,save_every:int,gpu_id=0):
        #"""
        # This function fine tunes BERT on a single GPU.
        #:param epochs: (int) number of total epochs for training 
        #:params bs: (int) batch size 
        #:params lr: (float) Learning rate
        #:params save_every: (int) Model will be saved for every certain number of epochs. 
        #    Example. If save_every=2, model will be saved after 2nd,4th,6th... epoch.
        #    Note : Final model will be always saved.
        #:params gpu_id: (int) index of device you want to use. 
        #"""
        model=CustomClassificationModel(checkpoint=self.checkpoint,num_class=self.num_class)

        # Save learning hyperparamters in a dictionary.
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='BERT_1gpu')

        # Create an instance from TrianerSingle class
        BERTTrainerSingle=TrainerSingle(gpu_id=gpu_id,
                                        model=model,
                                        trainset=self.train_dataset,
                                        testset=self.test_dataset,
                                        valset=self.val_dataset,
                                        num_class=self.num_class,
                                        const=const)

        BERTTrainerSingle.train(max_epochs=const['total_epochs'])
        BERTTrainerSingle.test()

    def BERT_DDP(self,rank:int,
                world_size:int,epochs:int,bs:int,lr:int
                ,save_every:int):
        #"""
        #This set up environment for distributed learning.

        #:param rank: (int) current working GPU index
        #:param world_size: (int) total number of GPUs available 
        #:param epochs: (int) number of total epochs
        #:param bs: (int) batch size
        #:param lr: (float) learning rate
        #:param save_every: (int) Model will be saved for every {save_every} epochs.
        #"""

        # Prepare the model for classification problem 
        model=CustomClassificationModel(checkpoint=self.checkpoint,num_class=self.num_class)

        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f'BERT_{world_size}gpus')
        
        ddp_setup(rank=rank,world_size=world_size)

        # Create an instance from the Trianer single class
        BERTTrainerDDP=TrainerDDP(gpu_id=rank,world_size=world_size,
                                    model=model,
                                    num_class=self.num_class,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    const=const)
        
        BERTTrainerDDP.train(max_epochs=const['total_epochs'])
        BERTTrainerDDP.test()

        destroy_process_group()

    def run_BERT_DDP(self,
                    world_size:int,
                    epochs:int,
                    bs:int,lr:int
                    ,save_every:int):
        """
        This function trains the model on multiple GPUs using pytorch DDP library.

        :param world_size: (int) total number of GPUs available 
        :param epochs: (int) number of total epochs
        :param bs: (int) batch size
        :param lr: (float) learning rate
        :param save_every: (int) Model will be saved for every ``save_every`` epochs.
        """

        # Prepare the model for classification problem 
        start=time.time()
        mp.spawn(self.BERT_DDP,args=(world_size,epochs,bs,lr,save_every),
                nprocs=world_size)
        end=time.time()
        print(f"RUNTIME for all processes : {end-start}")

    def run_BERT_torchrun(self,
                    world_size:int,
                    epochs:int,
                    bs:int,lr:int
                    ,save_every:int):
        model=CustomClassificationModel(checkpoint=self.checkpoint,num_class=self.num_class)
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f'BERT_{world_size}gpus')
        ddp_setup_torchrun()

        BERTTrainerMN=Trainer_multinode(world_size=world_size,
                                    model=model,
                                    num_class=self.num_class,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    const=const)
        BERTTrainerMN.train(max_epochs=const['total_epochs'])
        BERTTrainerMN.test()

        destroy_process_group()

class GPT():
    """
    Generative Pre-trained Transforemrs (GPT) module

    :param df: (DataFrame) raw data you want to use for fine tuning. Contains text(str) and labels(int).
    :param input_col: (str) name of the column that contains input text.
    :param output_col: (str) name of the column that contains output label.
    :param max_length: (int) maximum length of the input text. If the input exceeds maximum, the model will cut off.
    :param test_size: (float) portion of test data for model evalutation.
    :param val_size: (float) portion of validation data for training evaluation. 
    :param seed: (int) random seed for train and test split. 
    """
    def __init__(self,data_path:str,input_col:str,output_col:str,num_class:int,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42,encoding='utf-8'): # Model name should be BERT,GPT or LLAMA
        
        D=data_processor(path=data_path,input_col=input_col,
                        output_col=output_col,encoding=encoding)
        self.df=pd.read_csv(data_path,encoding=encoding)
        self.df=D.label_converter()
        self.checkpoint='gpt2' # model checkpoint
        self.X=self.df[input_col] 
        self.y=self.df[output_col] 
        self.num_class=num_class
        self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset(checkpoint=self.checkpoint)

    def run_GPT(self,epochs:int,bs:int,lr:float,save_every:int,gpu_id=0):
        #"""
        # This function fine tunes GPT2 on a single GPU.
        #:param epochs: (int) number of total epochs for training 
        #:params bs: (int) batch size 
        #:params lr: (float) Learning rate
        #:params save_every: (int) Model will be saved for every certain number of epochs. 
        #    Example. If save_every=2, model will be saved after 2nd,4th,6th... epoch.
        #    # Note : Final model will be always saved.
        #:params gpu_id: (int) index of device you want to use. 
        #"""

        model=CustomClassificationModel(checkpoint=self.checkpoint,num_class=self.num_class)
         
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='GPT2_1gpu')

        GPTTrainerSingle=TrainerSingle(gpu_id=gpu_id,
                                        model=model,
                                        trainset=self.train_dataset,
                                        testset=self.test_dataset,
                                        valset=self.val_dataset,
                                        num_class=self.num_class,
                                        const=const)

        GPTTrainerSingle.train(max_epochs=const['total_epochs'])
        GPTTrainerSingle.test()

    def GPT_DDP(self,rank:int,world_size:int, 
            epochs:int,bs:int,lr:float,
            save_every:int):
        #"""
        #This set up environment for distributed learning.

        #:param rank: (int) current working GPU index
        #:param world_size: (int) total number of GPUs available 
        #:param epochs: (int) number of total epochs
        #:param bs: (int) batch size
        #:param lr: (float) learning rate
        #:param save_every: (int) Model will be saved for every {save_every} epochs.
        #"""
        model=CustomClassificationModel(checkpoint=self.checkpoint,num_class=self.num_class)
         

        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f'GPT2_{world_size}gpus')
        
        ddp_setup(rank=rank,world_size=world_size)

        GPTTrainerDDP=TrainerDDP(gpu_id=rank,world_size=world_size,
                                        model=model,
                                        trainset=self.train_dataset,
                                        testset=self.test_dataset,
                                        valset=self.val_dataset,
                                        num_class=self.num_class,
                                        const=const)
        
        GPTTrainerDDP.train(max_epochs=const['total_epochs'])
        GPTTrainerDDP.test()
        
        destroy_process_group()

    def run_GPT_DDP(self,
            world_size:int, 
            epochs:int,bs:int,lr:float,
            save_every:int):
        """
        This function trains the model on multiple GPUs using pytorch DDP library.

        :param world_size: (int) total number of GPUs available 
        :param epochs: (int) number of total epochs
        :param bs: (int) batch size
        :param lr: (float) learning rate
        :param save_every: (int) Model will be saved for every {save_every} epochs.
        """
        start=time.time()
        mp.spawn(self.GPT_DDP,args=(world_size,epochs,bs,lr,save_every),
                nprocs=world_size)
        end=time.time()
        print(f"RUNTIME for all processes : {end-start}")

    def run_GPT_torchrun(self,
                    world_size:int,
                    epochs:int,
                    bs:int,lr:int
                    ,save_every:int):
        model=CustomClassificationModel(checkpoint=self.checkpoint,num_class=self.num_class)
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f'GPT_{world_size}gpus')
        ddp_setup_torchrun()

        GPTTrainerMN=Trainer_multinode(world_size=world_size,
                                    model=model,
                                    num_class=self.num_class,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    const=const)
        GPTTrainerMN.train(max_epochs=const['total_epochs'])
        GPTTrainerMN.test()

        destroy_process_group()

class Llama():
    """
    Large Language Model Meta AI(Llama) module 

    :param df: (DataFrame) raw data you want to use for fine tuning. Contains text(str) and labels(int).
    :param input_col: (str) name of the column that contains input text.
    :param output_col: (str) name of the column that contains output label.
    :param max_length: (int) maximum length of the input text. If the input exceeds maximum, the model will cut off.
    :param test_size: (float) portion of test data for model evalutation.
    :param val_size: (float) portion of validation data for training evaluation. 
    :param seed: (int) random seed for train and test split. 
    """
    def __init__(self,data_path:str,input_col:str,output_col:str,num_class:int,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42,encoding='utf-8'): # Model name should be BERT,GPT or LLAMA

        D=data_processor(path=data_path,input_col=input_col,
                         output_col=output_col,encoding=encoding)
        self.df=pd.read_csv(data_path,encoding=encoding) # Data we want to fine tune the model with. 
        self.df=D.label_converter()
        self.checkpoint='meta-llama/Llama-2-7b-hf' # model checkpoint
        self.X=self.df[input_col] # Raw input data
        self.y=self.df[output_col] # Raw output data
        self.num_class=num_class
        self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset(checkpoint=self.checkpoint)

    def run_LLAMA(self,epochs:int,bs:int,lr:float,save_every:int,gpu_id=0):
        #"""
        # This function fine tunes Llama on a single GPU.
        #:param epochs: (int) number of total epochs for training 
        #:params bs: (int) batch size 
        #:params lr: (float) Learning rate
        #:params save_every: (int) Model will be saved for every certain number of epochs. 
        #    Example. If save_every=2, model will be saved after 2nd,4th,6th... epoch.
        #    # Note : Final model will be always saved.
        #:params gpu_id: (int) index of device you want to use. 
        #"""
        PEFT=peft(checkpoint=self.checkpoint) # Create an instance
        model=PEFT.model # Load the quantized model
        tokenizer=PEFT.tokenizer # Load the tokenizer
        
        model=PEFTClassificationModel(model=model,num_class=self.num_class)
        print(model)

        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='Llama')
        
        trainer=TrainerSingle(gpu_id=gpu_id,model=model,
                            trainset=self.train_dataset,
                            testset=self.test_dataset,
                            valset=self.val_dataset,
                            num_class=self.num_class,
                            const=const)
        
        trainer.train(const['total_epochs'])
        trainer.test()