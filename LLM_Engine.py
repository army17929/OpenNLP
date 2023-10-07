from Trainer import ddp_setup,dataloader_ddp,TrainerDDP
from Trainer import prepare_const,dataloader_single,TrainerSingle
from ModelCustomize import load_model,create_bnb_config,GPTCustomClassificationModel, CustomClassificationModel
from Dataset_generation import prepare_dataset,prepare_dataset_BERT,load_preprocessed_nuclear_data
from transformers import AutoModel
import os 
import time
from pathlib import Path
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BERT():
    def __init__(self,df,input_col:str,output_col:str,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42): # Model name should be BERT,GPT or LLAMA
        self.df=df # Data we want to fine tune the model with. 
        self.checkpoint='bert-base-uncased' # model checkpoint
        self.X=df[input_col] # Raw input data
        self.y=df[output_col] # Raw output data
        self.train_dataset,self.test_dataset,self.val_dataset=prepare_dataset_BERT(X=self.X,
                                                                    y=self.y,test_size=test_size,
                                                                   val_size=val_size,checkpoint=self.checkpoint,
                                                                    seed=seed,max_length=max_length)
    
    def run_BERT(self,epochs:int,bs:int,lr:float,save_every:int):

        # Dataset is ready, let us prepare the model 
        model=CustomClassificationModel(
            checkpoint=AutoModel.from_pretrained(self.checkpoint),
                                        num_class=3)
        print(model)

        # Let's specify learning parameters.
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='BERT')

        # Load the data on the dataloader
        train_dataloader,test_dataloader,val_dataloader=dataloader_single(trainset=self.train_dataset,
                                                        testset=self.test_dataset,
                                                        valset=self.val_dataset,
                                                        bs=const['batch_size'])
        # Create an instance from the Trianer single class
        BERTTrainerSingle=TrainerSingle(gpu_id=1,
                                        model=model,
                                        trainloader=train_dataloader,
                                        testloader=test_dataloader,
                                        valloader=val_dataloader,
                                        const=const)

        start=time.time()
        BERTTrainerSingle.train(max_epochs=const['total_epochs'])
        BERTTrainerSingle.test(final_model_path=Path(f"./trained_{const['model_name']}/Nuclear_epoch{const['total_epochs']-1}.pt"))
        end=time.time()
        print(f'RUNTIME : {end-start}')

    def BERT_DDP(self,rank:int,
                world_size:int,epochs:int,bs:int,lr:int
                ,save_every:int):

        # Dataset is ready, let us prepare the model 
        model=CustomClassificationModel(
            checkpoint=AutoModel.from_pretrained(self.checkpoint),
                                        num_class=3)
        print(model)

        # Let's specify out learning parameters.
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='BERT_DDP')
        ddp_setup(rank,world_size)
        # Load the data on the dataloader
        train_dataloader,test_dataloader,val_dataloader,sampler_train,sampler_val=dataloader_ddp(trainset=self.train_dataset,
                                                        testset=self.test_dataset,
                                                        valset=self.val_dataset,
                                                        bs=const['batch_size'])
        # Create an instance from the Trianer single class
        BERTTrainerDDP=TrainerDDP(gpu_id=rank,
                                    model=model,
                                    trainloader=train_dataloader,
                                    testloader=test_dataloader,
                                    valloader=val_dataloader,
                                    sampler_train=sampler_train,
                                    sampler_val=sampler_val,
                                    const=const)
        
        BERTTrainerDDP.train(max_epochs=const['total_epochs'])
        BERTTrainerDDP.test(final_model_path=f"./trained_{const['model_name']}/Nuclear_epoch{const['total_epochs']-1}.pt")

        destroy_process_group()

    def run_BERT_DDP(self,
                    world_size:int,
                    epochs:int,
                    bs:int,lr:int
                    ,save_every:int):
        start=time.time()
        mp.spawn(self.BERT_DDP,args=(world_size,epochs,bs,lr,save_every),
                nprocs=world_size)
        end=time.time()
        print(f"RUNTIME : {end-start}")

class GPT():
    def __init__(self,df,input_col:str,output_col:str,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42): # Model name should be BERT,GPT or LLAMA
        self.df=df # Data we want to fine tune the model with. 
        self.checkpoint='gpt2' # model checkpoint
        self.X=df[input_col] # Raw input data
        self.y=df[output_col] # Raw output data
        self.train_dataset,self.test_dataset,self.val_dataset=prepare_dataset(X=self.X,
                                                                    y=self.y,test_size=test_size,
                                                                   val_size=val_size,checkpoint=self.checkpoint,
                                                                    seed=seed,max_length=max_length)
        
    def run_GPT(self,epochs:int,bs:int,lr:float,save_every:int):
        # Dataset is ready, let us prepare the model 
        model=GPTCustomClassificationModel(
            checkpoint=AutoModel.from_pretrained(self.checkpoint),
                                        num_class=3)
        print(model)

        # Let's specify out learning parameters.
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='GPT2')

        # Load the data on the dataloader
        train_dataloader,test_dataloader,val_dataloader=dataloader_single(trainset=self.train_dataset,
                                                        testset=self.test_dataset,
                                                        valset=self.val_dataset,
                                                        bs=const['batch_size'])
        # Create an instance from the Trianer single class
        GPTTrainerSingle=TrainerSingle(gpu_id=0,
                                        model=model,
                                        trainloader=train_dataloader,
                                        testloader=test_dataloader,
                                        valloader=val_dataloader,
                                        const=const)

        start=time.time()
        GPTTrainerSingle.train(max_epochs=const['total_epochs'])
        GPTTrainerSingle.test(final_model_path=Path(f"./trained_{const['model_name']}/Nuclear_epoch{const['total_epochs']-1}.pt"))
        end=time.time()
        print(f'RUNTIME : {end-start}')

    def GPT_DDP(self,rank:int,
                world_size:int, 
            epochs:int,bs:int,lr:float,
            save_every:int):

        # Dataset is ready, let us prepare the model 
        model=GPTCustomClassificationModel(
            checkpoint=AutoModel.from_pretrained(self.checkpoint),
                                        num_class=3)
        print(model)

        # Let's specify out learning parameters.
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='GPT2')
        ddp_setup(rank=rank,world_size=world_size)
        # Load the data on the dataloader
        train_dataloader,test_dataloader,val_dataloader,sampler_train,sampler_val=dataloader_ddp(trainset=self.train_dataset,
                                                        testset=self.test_dataset,
                                                        valset=self.val_dataset,
                                                        bs=const['batch_size'])
        # Create an instance from the Trianer single class
        GPTTrainerDDP=TrainerDDP(gpu_id=rank,
                                        model=model,
                                        trainloader=train_dataloader,
                                        testloader=test_dataloader,
                                        valloader=val_dataloader,
                                        sampler_val=sampler_val,
                                        sampler_train=sampler_train,
                                        const=const)

        GPTTrainerDDP.train(max_epochs=const['total_epochs'])
        GPTTrainerDDP.test(final_model_path=Path(f"./trained_{const['model_name']}/Nuclear_epoch{const['total_epochs']-1}.pt"))
        
        destroy_process_group()

    def run_GPT_DDP(self,
            world_size:int, 
            epochs:int,bs:int,lr:float,
            save_every:int):
        start=time.time()
        mp.spawn(self.GPT_DDP,args=(world_size,epochs,bs,lr,save_every),
                nprocs=world_size)
        end=time.time()
        print(f"RUNTIME : {end-start}")

class Llama():
    def __init__(self,df,input_col:str,output_col:str,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42): # Model name should be BERT,GPT or LLAMA
        self.df=df # Data we want to fine tune the model with. 
        self.checkpoint='meta-llama/Llama-2-7b-hf' # model checkpoint
        self.X=df[input_col] # Raw input data
        self.y=df[output_col] # Raw output data
        self.train_dataset,self.test_dataset,self.val_dataset=prepare_dataset(X=self.X,
                                                                    y=self.y,test_size=test_size,
                                                                   val_size=val_size,checkpoint=self.checkpoint,
                                                                    seed=seed,max_length=max_length)

    def run_LLAMA(self,epochs:int,bs:int,lr:float,save_every:int):

        # __Import the model and tokenizer__    
        bnb_config=create_bnb_config()
        model,tokenizer=load_model(gpu_id=1,checkpoint=self.checkpoint,bnb_config=bnb_config)
        model=CustomClassificationModel(checkpoint=model,num_class=3)
        print(model)

        # Create a dictionary that contains all parameters during the learning.
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='Llama')

        # Create a dataloader based on the batch size above.
        trainloader,testloader,valloader=dataloader_single(trainset=self.train_dataset,
                                                testset=self.test_dataset,
                                                valset=self.val_dataset,
                                                bs=const['batch_size'])
        # Create an instance from SingleGPUTrainer
        trainer=TrainerSingle(gpu_id=1,model=model,
                            trainloader=trainloader,
                            testloader=testloader,
                            valloader=valloader,
                            const=const)

        start=time.time()
        trainer.train(const['total_epochs'])
        trainer.test(final_model_path=f"./trained_{const['model_name']}/Nuclear_epoch4.pt")
        end=time.time()
        print(f'RUNTIME : {end-start} sec')