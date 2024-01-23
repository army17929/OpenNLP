from opennlp.trainer.trainer import TrainerFSDP,TrainerSingle,TrainerDDP,Trainer_multinode,prepare_const,ddp_setup,ddp_setup_torchrun
from opennlp.custommodel.model import CustomClassificationModel, peft, PEFTClassificationModel
from opennlp.preprocessing.data import data_processor
import os 
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group,init_process_group
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.distributed.fsdp import (MixedPrecision,
                                    ShardingStrategy,
                                    BackwardPrefetch,
                                    FullStateDictConfig,
                                    StateDictType)
from torch.distributed.fsdp.wrap import (transformer_auto_wrap_policy,
                                             size_based_auto_wrap_policy,
                                             _module_wrap_policy)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload,BackwardPrefetch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy,_module_wrap_policy

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

    def __init__(self,
                 user_split:bool,
                 input_col:str,output_col:str, num_class:int, 
                 data_path=None,
                 train_filepath=None,
                 test_filepath=None,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42,
                 encoding='utf-8'): 
        D=data_processor(user_split=user_split,
                         path=data_path,
                         train_filepath=train_filepath,
                         test_filepath=test_filepath,
                         input_col=input_col,
                         output_col=output_col,
                         encoding=encoding)

        self.checkpoint='bert-base-uncased' # model checkpoint
        self.num_class=num_class

        if not user_split:
            self.df=pd.read_csv(data_path,
                                encoding=encoding,
                                encoding_errors='ignore')
            self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset(checkpoint=self.checkpoint,
                                                                                    max_length=max_length,
                                                                                    test_size=test_size,
                                                                                    val_size=val_size,
                                                                                    seed=seed)
        if user_split:
            self.df_train=pd.read_csv(train_filepath,
                                      encoding=encoding,
                                      encoding_errors='ignore')
            self.df_test=pd.read_csv(test_filepath,
                                     encoding=encoding,
                                     encoding_errors='ignore')
            self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset(checkpoint=self.checkpoint,
                                                                                    max_length=max_length,
                                                                                    test_size=test_size,
                                                                                    val_size=val_size,
                                                                                    seed=seed)

    def zeroshot_BERT(self,bs:int):
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        model.to('cuda:0')
        y_pred=[]
        y_true=[]
        # Prepare the dataloader 
        testloader=DataLoader(self.test_dataset,
                              batch_size=bs,
                              shuffle=False)
        with torch.no_grad():
            for input,mask,tgt in testloader:
                input=input.to('cuda:0')
                mask=mask.to('cuda:0')
                tgt=tgt.to('cuda:0')
                out=model(input,mask)
                pred=torch.argmax(out,dim=1)
                y_pred.extend(pred.tolist())
                y_true.extend(tgt.tolist())
                if self.num_class==2:
                    out=torch.argmax(out,dim=1)
            result=classification_report(y_pred=y_pred,
                                         y_true=y_true)
            print(result)

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

    def run_BERT_DDP_torchrun(self,
                world_size:int,
                epochs:int,
                bs:int,
                lr:int
                ,save_every:int):
        """
        This set up environment for distributed learning when using torchrun.
        For users who do not use  torchrun can execute multiprocessing on their own.
        Nevertheless, this software recommends to use torchrun.
        Torchrun is more stable and less complexed.

        :param rank: (int) current working GPU index
        :param world_size: (int) total number of GPUs available 
        :param epochs: (int) number of total epochs
        :param bs: (int) batch size
        :param lr: (float) learning rate
        :param save_every: (int) Model will be saved for every {save_every} epochs.
        """
        print('CURRENT DEVICE',torch.cuda.current_device())
        # Prepare the model for classification problem 
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f'BERT_DDP_{world_size}gpus')
        
        ddp_setup_torchrun()

        # Create an instance from the Trianer single class
        rank=int(os.environ['LOCAL_RANK'])
        BERTTrainerDDP=TrainerDDP(gpu_id=rank,
                                  world_size=world_size,
                                    model=model,
                                    num_class=self.num_class,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    const=const)
        
        BERTTrainerDDP.train(max_epochs=const['total_epochs'])
        BERTTrainerDDP.test()

        destroy_process_group()
    
    def BERT_DDP(self,
                 rank:int,
                 world_size:int,
                 epochs:int,
                 bs:int,
                 lr:float,
                 save_every:int):
        """
        This is a function that will be multiprocessed, when the user 
        try to run DDP without torchrun.
        """
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        const=prepare_const(num_epochs=epochs,
                            batch_size=bs,
                            lr=lr,
                            save_every=save_every,
                            model_name=f'BERT_{world_size}gpus')
        ddp_setup(rank=rank,world_size=world_size)

        BERTTrainerDDP=TrainerDDP(gpu_id=rank,
                                world_size=world_size,
                                model=model,
                                trainset=self.train_dataset,
                                testset=self.test_dataset,
                                valset=self.val_dataset,
                                num_class=self.num_class,
                                const=const)
        BERTTrainerDDP.train(max_epochs=const['total_epochs'])
        BERTTrainerDDP.test()

    def run_BERT_DDP(self,
                    world_size:int,
                    epochs:int,
                    bs:int,lr:int
                    ,save_every:int):
        """
        This function trains the model on multiple GPUs using pytorch DDP library
        without torchrun. This function directly

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
    
    def run_BERT_FSDP(self,
                    world_size:int,
                    epochs:int,bs:int,lr:float,
                    save_every:int,
                    min_num_params=int(1e2),
                    cpu_offload=False,
                    full_shard=True,
                    mixed_precision=False):
        """
        Note that this function can be only used with torchrun.
        """
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        const=prepare_const(num_epochs=epochs,
                            batch_size=bs,
                            lr=lr,
                            save_every=save_every,
                            model_name=f"BERT_FSDP_{world_size}gpus")

        ddp_setup_torchrun()

        BERTTrainerfsdp=TrainerFSDP(world_size=world_size,
                                    model=model,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    num_class=self.num_class,
                                    const=const,
                                    lr=lr,epochs=epochs,
                                    min_num_params=min_num_params,
                                    cpu_offload=cpu_offload,
                                    full_shard=full_shard,
                                    mixed_precision=mixed_precision)
        
        BERTTrainerfsdp.train()
        torch.cuda.empty_cache()

        BERTTrainerfsdp.test()
        destroy_process_group()

    def run_4bit_BERT(self,
                        epochs:int,
                        bs:int,
                        lr:float,
                        save_every:int,
                        gpu_id=0):
        PEFT=peft(checkpoint=self.checkpoint,
                  load_in_4bit=True) # Create an instance from peft
        quantized_model=PEFT.model # Load in 4bit model 
        model=PEFTClassificationModel(model=quantized_model,
                                      num_class=self.num_class)
        const=prepare_const(num_epochs=epochs,
                            batch_size=bs,
                            lr=lr,
                            save_every=save_every,
                            model_name='BERT_4bit_quantized')
        BERTTrainerSingle=TrainerSingle(gpu_id=gpu_id,
                                        model=model,
                                        trainset=self.train_dataset,
                                        testset=self.test_dataset,
                                        valset=self.val_dataset,
                                        num_class=self.num_class,
                                        const=const)
        BERTTrainerSingle.train(max_epochs=const['total_epochs'])
        BERTTrainerSingle.test()

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
    def __init__(self,
                 user_split:bool,
                 input_col:str,
                 output_col:str,
                 num_class:int,
                 data_path:str,
                 train_filepath:str,
                 test_filepath:str,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42,encoding='utf-8'): # Model name should be BERT,GPT or LLAMA
        
        D=data_processor(user_split=user_split,
                         path=data_path,
                         train_filepath=train_filepath,
                         test_filepath=test_filepath,
                         input_col=input_col,
                         output_col=output_col,
                         encoding=encoding)
        
        # self.df=D.label_converter()
        self.checkpoint='gpt2' # model checkpoint
        self.num_class=num_class
        if not user_split:
            self.df=pd.read_csv(data_path,
                                encoding=encoding,
                                encoding_errors='ignore')
            self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset(checkpoint=self.checkpoint,
                                                                                    max_length=max_length,
                                                                                    test_size=test_size,
                                                                                    val_size=val_size,
                                                                                    seed=seed)
        if user_split:
            self.df_train=pd.read_csv(train_filepath,
                                      encoding=encoding,
                                      encoding_errors='ignore')
            self.df_test=pd.read_csv(test_filepath,
                                     encoding=encoding,
                                     encoding_errors='ignore')
            self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset(checkpoint=self.checkpoint,
                                                                                    max_length=max_length,
                                                                                    test_size=test_size,
                                                                                    val_size=val_size,
                                                                                    seed=seed)
    def zeroshot_GPT(self,bs:int):
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        model.to('cuda:0')
        y_pred=[]
        y_true=[]
        testloader=DataLoader(self.test_dataset,
                              batch_size=bs,
                              shuffle=False)
        with torch.no_grad():
            for input,mask,tgt in testloader:
                input=input.to('cuda:0')
                mask=mask.to('cuda:0')
                tgt=tgt.to('cuda:0')
                out=model(input,mask)
                pred=torch.argmax(out,dim=1)
                y_pred.extend(pred.tolist())
                y_true.extend(tgt.tolist())
                if self.num_class==2:
                    out=torch.argmax(out,dim=1)
            result =classification_report(y_true=y_true,
                                          y_pred=y_pred)
            print(result)


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

    def run_GPT_DDP_torchrun(self,
                              world_size:int,
                              epochs:int,
                              bs:int,
                              lr:int,
                              save_every:int):
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f'GPT_DDP_{world_size}gpus')
        ddp_setup_torchrun()

        rank=int(os.environ['LOCAL_RANK'])
        GPTTrainerDDP=TrainerDDP(gpu_id=rank,world_size=world_size,
                                    model=model,
                                    num_class=self.num_class,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    const=const)
        
        GPTTrainerDDP.train(max_epochs=const['total_epochs'])
        GPTTrainerDDP.test()

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

    def run_GPT_FSDP(self,
                 world_size:int,
                      epochs:int,bs:int,lr:float,
                      save_every:int,
                      cpu_offload=False):
        """
        FSDP can only be used with torchrun.
        """
        print("multiprocessing starts...")
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        print('model loaded...')
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f"GPT_{world_size}gpus")
        # FSDP setup is same as DDP
        ddp_setup_torchrun()
        print("DDP setup completed...")

        GPTTrainerfsdp=TrainerFSDP(world_size=world_size,
                                    model=model,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    num_class=self.num_class,
                                    const=const,
                                    lr=lr,epochs=epochs,
                                    cpu_offload=cpu_offload)
        print('FSDP Trainer object initialized...')
        GPTTrainerfsdp.train()
        torch.cuda.empty_cache()
        GPTTrainerfsdp.test()
        
        destroy_process_group()

    def run_4bit_GPT(self,
                     epochs:int,
                     bs:int,
                     lr:float,
                     save_every:int,
                     gpu_id=0):
        PEFT=peft(checkpoint=self.checkpoint,
                  load_in_4bit=True)
        quantized_model=PEFT.model
        model=PEFTClassificationModel(model=quantized_model,
                                      num_class=self.num_class)
        const=prepare_const(num_epochs=epochs,
                            batch_size=bs,
                            lr=lr,
                            save_every=save_every,
                            model_name='GPT_4bit_quantized')
        GPTTrainerSingle=TrainerSingle(gpu_id=gpu_id,
                                       model=model,
                                       trainset=self.train_dataset,
                                       testset=self.test_dataset,
                                       valset=self.val_dataset,
                                       num_class=self.num_class,
                                       const=const)
        GPTTrainerSingle.train(max_epochs=const['total_epochs'])
        GPTTrainerSingle.test()

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
    def __init__(self,
                 user_split:bool,
                 input_col:str,output_col:str,num_class:str,
                 data_path=None,
                 train_filepath=None,
                 test_filepath=None,
                 max_length=128,test_size=0.2,val_size=0.1,seed=42,
                 encoding='utf-8'): # Model name should be BERT,GPT or LLAMA
        print("initializing Llama2...")
        D=data_processor(user_split=user_split,
                         path=data_path,
                         train_filepath=train_filepath,
                         test_filepath=test_filepath,
                         input_col=input_col,
                         output_col=output_col,
                         encoding=encoding)
        
        self.checkpoint='meta-llama/Llama-2-7b-hf' # model checkpoint
        self.num_class=num_class
        if not user_split:
            self.df=pd.read_csv(data_path,
                                encoding=encoding,
                                encoding_errors='ignore')
            self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset(checkpoint=self.checkpoint,
                                                                                    max_length=max_length,
                                                                                    test_size=test_size,
                                                                                    val_size=val_size,
                                                                                    seed=seed)
        if user_split:
            self.df_train=pd.read_csv(train_filepath,
                                      encoding=encoding,
                                      encoding_errors='ignore')
            self.df_test=pd.read_csv(test_filepath,
                                     encoding=encoding,
                                     encoding_errors='ignore')
            self.train_dataset,self.test_dataset,self.val_dataset=D.prepare_dataset(checkpoint=self.checkpoint,
                                                                                    max_length=max_length,
                                                                                    test_size=test_size,
                                                                                    val_size=val_size,
                                                                                    seed=seed)

    def zeroshot_LLAMA(self,bs:int):
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        model.to('cuda:0')
        y_pred=[]
        y_true=[]
        # Prepare the dataloader 
        testloader=DataLoader(self.test_dataset,
                              batch_size=bs,
                              shuffle=False)
        with torch.no_grad():
            for input,mask,tgt in testloader:
                input=input.to('cuda:0')
                mask=mask.to('cuda:0')
                tgt=tgt.to('cuda:0')
                out=model(input,mask)
                pred=torch.argmax(out,dim=1)
                y_pred.extend(pred.tolist())
                y_true.extend(tgt.tolist())
                if self.num_class==2:
                    out=torch.argmax(out,dim=1)
            result=classification_report(y_pred=y_pred,
                                         y_true=y_true)
            print(result)

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
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        
        print(model)

        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name='Llama')
        
        Llamatrainer=TrainerSingle(gpu_id=gpu_id,model=model,
                            trainset=self.train_dataset,
                            testset=self.test_dataset,
                            valset=self.val_dataset,
                            num_class=self.num_class,
                            const=const)
        
        Llamatrainer.train(const['total_epochs'])
        Llamatrainer.test()

    def run_LLAMA_DDP_torchrun(self,
                    world_size:int,
                    epochs:int,
                    bs:int,
                    lr:int,
                    save_every:int):
        model=CustomClassificationModel(checkpoint=self.checkpoint,
                                        num_class=self.num_class)
        print(model)
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f'Llama_DDP_{world_size}gpus')
        ddp_setup_torchrun()
        rank=int(os.environ['LOCAL_RANK'])
        LlamaTrainer=TrainerDDP(gpu_id=rank,
                                world_size=world_size,
                                model=model,
                                num_class=self.num_class,
                                trainset=self.train_dataset,
                                testset=self.test_dataset,
                                valset=self.val_dataset,
                                const=const)
        
        LlamaTrainer.train(max_epochs=const['total_epochs'])
        LlamaTrainer.test()

        destroy_process_group()

    def LLAMA_DDP(self,rank:int,
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
        #model=peft(checkpoint=self.checkpoint).model # Create an instance
        #print("Converting the model into 4bit...")
        #model=PEFTClassificationModel(model=model,num_class=self.num_class).to(rank)
        #print(f'model replicated on cuda:{rank}')
        from transformers import AutoModel
        model=CustomClassificationModel(checkpoint='meta-llama/Llama-2-7b-hf',
                                        num_class=self.num_class).half().to(rank)

        print(model)
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f'Llama_{world_size}gpus')

        ddp_setup(rank=rank,world_size=world_size)
        print("DDP setup completed...")
        # Create an instance from the Trianer single class
        llamaTrainerDDP=TrainerDDP(gpu_id=rank,world_size=world_size,
                                    model=model,
                                    num_class=self.num_class,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    const=const)
        
        llamaTrainerDDP.train(max_epochs=const['total_epochs'])
        llamaTrainerDDP.test()

        destroy_process_group()

    def run_LLAMA_DDP(self,
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
        print("start multiprocessing...")
        mp.spawn(self.LLAMA_DDP,args=(world_size,epochs,bs,lr,save_every),
                nprocs=world_size)
        end=time.time()
        print(f"RUNTIME for all processes : {end-start}")

    def run_LLAMA_FSDP(self,
                      world_size:int,
                      epochs:int,bs:int,lr:float,
                      save_every:int,
                      min_num_params=int(1e2),
                      cpu_offload=False,
                      full_shard=True,
                      mixed_precision=False):
        
        model=CustomClassificationModel(checkpoint='meta-llama/Llama-2-7b-hf',
                                        num_class=self.num_class).half()
        
        const=prepare_const(num_epochs=epochs,batch_size=bs,
                            lr=lr,save_every=save_every,
                            model_name=f"LLAMA_{world_size}gpus")

        ddp_setup_torchrun()
        print("DDP setup completed...")

        llamaTrainerfsdp=TrainerFSDP(world_size=world_size,
                                    model=model,
                                    trainset=self.train_dataset,
                                    testset=self.test_dataset,
                                    valset=self.val_dataset,
                                    num_class=self.num_class,
                                    const=const,
                                    lr=lr,epochs=epochs,
                                    min_num_params=min_num_params,
                                    full_shard=full_shard,
                                    mixed_precision=mixed_precision,
                                    cpu_offload=cpu_offload)
        
        llamaTrainerfsdp.train()
        torch.cuda.empty_cache()

        llamaTrainerfsdp.test()        
        destroy_process_group()

    def run_4bit_LLAMA(self,
                       epochs:int,
                       bs:int,
                       lr:float,
                       save_every:int,
                       gpu_id=0):
        if gpu_id!=0:
            RuntimeError("Quantized model cannot be parallelized")
        PEFT=peft(checkpoint=self.checkpoint,
                  load_in_4bit=True)
        quantized_model=PEFT.model
        model=PEFTClassificationModel(model=quantized_model,
                                      num_class=self.num_class)
        const=prepare_const(num_epochs=epochs,
                            batch_size=bs,
                            lr=lr,
                            save_every=save_every,
                            model_name='Llama_4bit_quantized')
        LlamaTrainerSingle=TrainerSingle(gpu_id=gpu_id,
                                         model=model,
                                         trainset=self.train_dataset,
                                         testset=self.test_dataset,
                                         valset=self.val_dataset,
                                         num_class=self.num_class,
                                         const=const)
        LlamaTrainerSingle.train(max_epochs=const['total_epochs'])
        LlamaTrainerSingle.test()
