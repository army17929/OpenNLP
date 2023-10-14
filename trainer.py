from pathlib import Path
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import Tensor
from typing import Tuple
import torchmetrics
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
import bitsandbytes as bnb
import os
from sklearn.metrics import classification_report,confusion_matrix
import time

class TrainerSingle:
    """
    Trainer for Single GPU module

    :param gpu_id: (int) index of GPU you want to use for training. 0 if you have only 1 GPU.
    :param model: (object) model for training 
    :param trainset: (TensorDataset) tensor dataset for training
    :param testset: (TensorDataset) tensor dataset for test
    :param valset: (TensorDataset) tensor dataset for validation
    :param const: (Dict) Dictionary that contains learning hyperparameters 
    """
    def __init__(self,gpu_id:int,model:nn.Module,
                 trainset:Dataset,
                 testset:Dataset,
                 valset:Dataset,const,
                 num_class:int,world_size=1):
        self.gpu_id=gpu_id # Current working thread
        self.const=const # Learning parameters 
        self.batch_size=self.const['batch_size']
        self.model=model.to(self.gpu_id) # Send the model to GPU
        self.num_class=num_class
        self.trainset=trainset 
        self.testset=testset
        self.valset=valset
        self.criterion=nn.CrossEntropyLoss()
        self.optimizer=bnb.optim.AdamW8bit(self.model.parameters(),
                                 lr=self.const['lr'])
        self.lr_scheduler=optim.lr_scheduler.StepLR(
            self.optimizer,self.const['lr_step_size']
        )
        # binary classification
        if self.num_class==2:
            self.train_acc=torchmetrics.Accuracy(
                task="multiclass",num_classes=self.num_class,average="micro"
            ).to(self.gpu_id)
            self.val_acc=torchmetrics.Accuracy(
            task="multiclass",num_classes=self.num_class,average="micro"
            ).to(self.gpu_id)
            self.test_acc=torchmetrics.Accuracy(
                task="multiclass",num_classes=self.num_class,average="micro"
            ).to(self.gpu_id)
        # Multiclass classification
        else:
            self.train_acc=torchmetrics.Accuracy(
                task="multiclass",num_classes=self.num_class,average="micro"
            ).to(self.gpu_id)
            self.val_acc=torchmetrics.Accuracy(
            task="multiclass",num_classes=self.num_class,average="micro"
            ).to(self.gpu_id)
            self.test_acc=torchmetrics.Accuracy(
                task="multiclass",num_classes=self.num_class,average="micro"
            ).to(self.gpu_id)
        self.train_acc_array=torch.zeros(self.const['total_epochs'])
        self.val_acc_array=torch.zeros(self.const['total_epochs'])
        self.train_loss_array=torch.zeros(self.const['total_epochs'])
        self.val_loss_array=torch.zeros(self.const['total_epochs'])
        self.trainloader,self.testloader,self.valloader=self.dataloader_single()
        self.best_model_path=''
        self.world_size=1
        self.runtime=0
    
    def dataloader_single(self)->Tuple[DataLoader,DataLoader,DataLoader]:
        #"""
        #Dataloader for single GPU
        #This function will generate dataloaders from datasets, based on the batch size. 
        #"""
        trainloader=DataLoader(self.trainset,batch_size=self.batch_size,shuffle=True,num_workers=1)
        testloader=DataLoader(self.testset,batch_size=self.batch_size,shuffle=False,num_workers=1)
        valloader=DataLoader(self.valset,batch_size=self.batch_size,shuffle=False,num_workers=1)
        return trainloader,testloader,valloader

    def _run_batch(self,src:list,tgt:Tensor)->float:
        # Running each batch
        self.optimizer.zero_grad() 
        out=self.model(src[0],src[1])
        loss=self.criterion(out,tgt)
        loss.backward()
        self.optimizer.step()

        self.train_acc.update(out,tgt)
        self.val_acc.update(out,tgt)
        return loss.item()
    
    def _run_epoch(self,epoch:int):
        # Running each epoch
        self.model.train()
        loss=0.0
        val_loss=0.0
        self.train_acc.reset()
        for input,mask,tgt in self.trainloader:
            input=input.to(self.gpu_id)
            mask=mask.to(self.gpu_id)
            tgt=tgt.to(self.gpu_id)
            src=[input,mask] 
            loss_batch=self._run_batch(src,tgt)
            loss+=loss_batch
        self.lr_scheduler.step()
        self.train_acc_array[epoch]=self.train_acc.compute().item()
        self.train_loss_array[epoch]=loss/len(self.trainloader)
        
        # Run on the validation set
        self.model.eval()
        self.val_acc.reset()
        for input,mask,tgt in self.valloader:
            input=input.to(self.gpu_id)
            mask=mask.to(self.gpu_id)
            tgt=tgt.to(self.gpu_id)
            src=[input,mask] 
            loss_batch_val=self._run_batch(src,tgt)
            val_loss+=loss_batch_val
        self.lr_scheduler.step()
        
        # Save validation acc
        self.val_acc_array[epoch]=self.val_acc.compute().item()
        self.val_loss_array[epoch]=val_loss/len(self.valloader)
        print(f"{'-'*90}\n [GPU{self.gpu_id}] Epoch {epoch+1:2d} \
              | Batchsize: {self.const['batch_size']} | Steps : {len(self.trainloader)} \
                LR :{self.optimizer.param_groups[0]['lr']:.2f}, \
                Train_Loss: {loss/len(self.trainloader):.2f}\
                Val_Loss: {val_loss/len(self.valloader):.2f}\
                Training_Acc: {100*self.train_acc.compute().item():.2f}% \
                Val_Acc: {100*self.val_acc.compute().item():.2f}",flush=True)
        
        if epoch==self.const['total_epochs']-1 : # Save the loss and acc plot at the last epoch
            # Loss plot
            plt.clf() # Initialize the plot
            plt.figure(figsize=(14,10))
            plt.subplot(1,2,1)
            plt.subplots_adjust(wspace=0.2)
            plt.plot(np.arange(1,self.const['total_epochs']+1),
                     self.train_loss_array,label='Train Loss')
            plt.plot(np.arange(1,self.const['total_epochs']+1),
                     self.val_loss_array,label='Val Loss')
            plt.xlabel('Epoch')
            plt.title(f"Train & Val Loss - {self.const['model_name']}")
            plt.legend()
            plt.grid()
            # Accuracy plot
            plt.subplot(1,2,2)
            plt.plot(np.arange(1,self.const['total_epochs']+1),
                     self.train_acc_array,label='Train Acc')
            plt.plot(np.arange(1,self.const['total_epochs']+1),
                     self.val_acc_array,label='Val Acc')
            plt.xlabel('Epoch')
            plt.title(f"Train & Val Acc -  {self.const['model_name']}")
            plt.legend()
            plt.grid()
            save_dir=f"./Results/{self.const['model_name']}_epoch_{self.const['total_epochs']}_batch_{self.const['batch_size']}_{self.world_size}gpu"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
            plt.savefig(f"{save_dir}/Trainingplot_{self.const['model_name']}_epoch_{self.const['total_epochs']}_bs_{self.const['batch_size']}.png")

    def _save_checkpoint(self,epoch:int,model_name:str):
        """
        Save checkpoint function 
        This function will save the model during the training. 
        You can load the model from the checkpoint later.
        """
        checkpoint=self.model.state_dict()
        model_path=self.const["trained_models"]/f"{model_name}_epoch{epoch+1}.pt"
        torch.save(checkpoint,model_path)
    
    def train(self,max_epochs:int):
        # Comprehensive training function
        self.model.train()
        start=time.time()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            print(self.val_acc_array)
            if epoch==0: # First epoch, save it anyways.
                self._save_checkpoint(epoch,model_name=f"best_{self.const['model_name']}")
            if self.val_acc_array[epoch]>self.val_acc_array[epoch-1]:
                """
                If the model's performance is improved than the previous epoch, save it
                """
                self._save_checkpoint(epoch,model_name=f"best_{self.const['model_name']}")
            if epoch % self.const['save_every']==0:
                self._save_checkpoint(epoch,model_name=f"{self.const['model_name']}")
        end=time.time()
        runtime=end-start
        print(f"TRAINING RUNTIME of {self.const['model_name']} : {end-start:2f} sec")
        #self._save_checkpoint(epoch=max_epochs-1,model_name=f"last_{self.const['model_name']}") # save the last epoch
        #self._save_checkpoint(epoch=torch.argmax(self.val_acc_array),model_name=f"Best_{self.const['model_name']}")
        self.best_model_path=self.const["trained_models"]/f"best_{self.const['model_name']}_epoch{torch.argmax(self.val_acc_array)+1}.pt" 
        self.runtime=runtime

    def test(self):
        # Model evaluation function
        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"TEST MODEL : {self.best_model_path}")
        self.model.eval()
        y_true=[]
        y_pred=[]
        with torch.no_grad():
            for input,mask,tgt in self.testloader:
                input=input.to(self.gpu_id)
                mask=mask.to(self.gpu_id)
                tgt=tgt.to(self.gpu_id)
                out=self.model(input,mask)
                pred=torch.argmax(out,dim=1)
                y_pred.extend(pred.tolist())
                y_true.extend(tgt.tolist())
                self.test_acc.update(out,tgt)
        print(f"[GPU{self.gpu_id}] \
              Test Acc: {100 * self.test_acc.compute().item():.4f}%")
        result=classification_report(y_true,y_pred)
        print(result)
        # Save the confusion matrix
        if self.num_class==2:
            binary_metrics_generator(y_true=y_true,y_pred=y_pred,
                     save_dir=f"/{self.const['model_name']}_epoch_{self.const['total_epochs']}_batch_{self.const['batch_size']}_1gpu",
                     model_name=f"{self.const['model_name']}",runtime=self.runtime)
        else:
            metrics_generator(y_true=y_true,y_pred=y_pred,
                     save_dir=f"/{self.const['model_name']}_epoch_{self.const['total_epochs']}_batch_{self.const['batch_size']}_1gpu",
                     model_name=f"{self.const['model_name']}",runtime=self.runtime)

class TrainerDDP(TrainerSingle):
    """
    Trainer module for Multiple GPUs

    :param gpu_id: (int) index of current device
    :param model: (object) model for training  
    :param trainset: (TensorDataset) tensor dataset for training
    :param testset: (TensorDataset) tensor dataset for test
    :param valset: (TensorDataset) tensor dataset for validation
    :param const: (Dict) Dictionary that contains learning hyperparameters 
    """
    def __init__(
            self, world_size:int, 
            gpu_id:int, model:nn.Module,
            trainset:Dataset,testset:Dataset,
            valset:Dataset, num_class:int,
            const
    )->None:
        super().__init__(gpu_id,model,trainset,
                         testset,valset,const,num_class)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        # Wrap the model with DDP
        self.model=DDP(self.model,device_ids=[gpu_id],find_unused_parameters=True)
        self.trainloader,self.testloader,self.valloader,self.sampler_train,self.sampler_val= self.dataloader_ddp()
        self.best_model_path=''
        self.num_class=num_class
        self.world_size=world_size

    def dataloader_ddp(self)-> Tuple[DataLoader,DataLoader,DataLoader,DistributedSampler,DistributedSampler]:
        #"""
        #Dataloader function for multiple GPUs. 
        #This function will distribute the data for each device. 
        #"""
        sampler_train=DistributedSampler(self.trainset)
        sampler_val=DistributedSampler(self.valset)
        trainloader=DataLoader(self.trainset,batch_size=self.const['batch_size'],
                            shuffle=False,sampler=sampler_train,
                            num_workers=2)
        testloader=DataLoader(self.testset,batch_size=self.const['batch_size'])
        valloader=DataLoader(self.valset,batch_size=self.const['batch_size'],
                            shuffle=False,sampler=DistributedSampler(self.valset,shuffle=False),
                            num_workers=2)
        return trainloader,testloader,valloader,sampler_train,sampler_val

    def _save_checkpoint(self,epoch:int,model_name:str):
        checkpoint=self.model.module.state_dict()
        model_path=self.const["trained_models"]/f"{model_name}_epoch{epoch+1}.pt"
        torch.save(checkpoint,model_path)
    
    def train(self,max_epochs:int):
        self.model.train()
        start=time.time()
        for epoch in range(max_epochs):
            self.sampler_train.set_epoch(epoch)
            self.sampler_val.set_epoch(epoch)
            self._run_epoch(epoch)
            if epoch==0: # First epoch, save it anyways.``
                self._save_checkpoint(epoch,model_name=f"best_{self.const['model_name']}")
            if self.val_acc_array[epoch]>self.val_acc_array[epoch-1]:
                self._save_checkpoint(epoch,model_name=f"best_{self.const['model_name']}")
            if epoch%self.const["save_every"]==0:
                self._save_checkpoint(epoch,model_name=f"{self.const['model_name']}")
        end=time.time()
        runtime=end-start
        print(f"RUNTIME of process{self.gpu_id} / {self.const['model_name']} : {end-start:2f} sec")
        #self._save_checkpoint(epoch=max_epochs-1,model_name=f"Last_{self.const['model_name']}")
        #self._save_checkpoint(epoch=torch.argmax(self.val_acc_array),model_name=f"Best_{self.const['model_name']}")
        self.best_model_path=self.const["trained_models"]/f"best_{self.const['model_name']}_epoch{torch.argmax(self.val_acc_array)+1}.pt" 
        self.runtime=runtime

    def test(self):
        self.model.module.load_state_dict(
            torch.load(self.best_model_path,map_location="cpu"))
        self.model.eval()
        y_true=[]
        y_pred=[]
        # Eval mode
        with torch.no_grad():
            for input,mask,tgt in self.testloader:
                input=input.to(self.gpu_id)
                mask=mask.to(self.gpu_id)
                tgt=tgt.to(self.gpu_id)
                out=self.model(input,mask)
                pred=torch.argmax(out,dim=1)
                y_pred.extend(pred.tolist())
                y_true.extend(tgt.tolist())
                self.test_acc.update(out,tgt)

        print(f"[GPU{self.gpu_id} Test Acc : {100*self.test_acc.compute().item():4f}%]")
        result=classification_report(y_true,y_pred)
        print(result)
        if self.num_class==2:
            binary_metrics_generator(y_true=y_true,y_pred=y_pred,
                        save_dir=f"/{self.const['model_name']}_epoch_{self.const['total_epochs']}_batch_{self.const['batch_size']}_{self.world_size}gpu",
                        model_name=f"{self.const['model_name']}",runtime=self.runtime)
        else:
            metrics_generator(y_true=y_true,y_pred=y_pred,
                        save_dir=f"/{self.const['model_name']}_epoch_{self.const['total_epochs']}_batch_{self.const['batch_size']}_{self.world_size}gpu",
                        model_name=f"{self.const['model_name']}",runtime=self.runtime)

def PlotTraining(history,savedir:str,model_name:str):
    # This function will generate training and validation plots, and save it in ``savedir`` 
    save_dir='./Results'+savedir
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    train_acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    # Plot Loss
    plt.figure(figsize=(14,10))
    plt.subplot(1,2,1)
    plt.plot(train_loss,label='Train_loss')
    plt.plot(val_loss,label='Val_loss')
    plt.legend()
    plt.grid()
    plt.title(f"Loss - {model_name}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Plot Acc
    plt.subplot(1,2,2)
    plt.plot(train_acc,label='Train_acc')
    plt.plot(val_acc,label='Val_acc')
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title(f"Acc - {model_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    plt.savefig(f"{save_dir}/Trainingplot_{model_name}")

def prepare_const(num_epochs:int,batch_size:int,
                  lr:float,save_every:int,model_name:str)->dict:
    # Constant hyperparameters preparation function 
    data_root=Path('data') # Data directory
    trained_models=Path(f'trained_{model_name}') # Model directory
    
    if not data_root.exists():
        data_root.mkdir()
    if not trained_models.exists():
        trained_models.mkdir()
    
    const=dict(data_root=data_root,
               trained_models=trained_models,
               total_epochs=num_epochs,
               batch_size=batch_size, lr=lr,
               momentum=0.9, lr_step_size=5,
               save_every=save_every,model_name=model_name)
    return const


def metrics_generator(y_true:list,y_pred:list,save_dir:str,model_name:str,runtime:float):
    #"""
    #Metrics generator function 
    #This function will generate confusion matrix and classification report
    #Plot and report are saved in ``savedir``
    #"""
    save_dir='./Results'+save_dir
    cm=confusion_matrix(y_true,y_pred)
    # Visualize the confusion matrix 
    plt.clf()
    plt.figure(figsize=(8,6))
    # Prepare the label that we want to put in each cell.
    cm_names=['True Neutral','False Pos','False Neg',
              'False Neutral','True Pos','False Neg',
              'False Neutral','False Pos','True Neg']
    cm_counts=[count for count in cm.flatten()]
    cm_percentages=["{0:.2%}".format(count) for count in cm.flatten()/np.sum(cm)]
    labels=[f"{v1}\n{v2}\n{v3}" for v1,v2,v3 in 
            zip(cm_names,cm_counts,cm_percentages)]
    labels=np.asarray(labels).reshape(3,3)
    sns.heatmap(data=cm,annot=labels,fmt='',cmap='Blues',
                xticklabels=['Neutral','Positive','Negative'],
                yticklabels=['Neutral','Positive','Negative'])
    plt.title(f"Confusion Matrix of {model_name}")        
    # Save the matrix
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,f"cm_{model_name}")
    plt.savefig(save_path)
    # Create the classification report 
    report=classification_report(y_true,y_pred)
    file=f"/{model_name}_classification_report.txt"
    with open(save_dir+file,'w') as f :
        f.write(report)
        f.write(f"RUNTIME {model_name} : {runtime}")

def binary_metrics_generator(y_true:list,y_pred:list,save_dir:str,model_name:str,runtime:float):
    #"""
    #Metrics generator function for bunary classification
    #This function will generate confusion matrix and classification report
    #Plot and report are saved in ``savedir``
    #"""
    save_dir='./Results'+save_dir
    cm=confusion_matrix(y_true,y_pred)
    # Visualize the confusion matrix 
    plt.clf()
    plt.figure(figsize=(8,6))
    
    # Prepare the label that we want to put in each cell.
    cm_names=['True Pos','False Neg',
              'False Pos','True Neg']
    cm_counts=[count for count in cm.flatten()]
    cm_percentages=["{0:.2%}".format(count) for count in cm.flatten()/np.sum(cm)]
    labels=[f"{v1}\n{v2}\n{v3}" for v1,v2,v3 in 
            zip(cm_names,cm_counts,cm_percentages)]
    labels=np.asarray(labels).reshape(2,2)
    sns.heatmap(data=cm,annot=labels,fmt='',cmap='Blues',
                xticklabels=['Positive','Negative'],
                yticklabels=['Positive','Negative'])
    plt.title(f"Confusion Matrix of {model_name}")        
    # Save the matrix
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,f"cm_{model_name}")
    plt.savefig(save_path)
    # Create the classification report 
    report=classification_report(y_true,y_pred)
    file=f"/{model_name}_classification_report.txt"
    with open(save_dir+file,'w') as f :
        f.write(report)
        f.write(f"RUNTIME {model_name} : {runtime}")

def ddp_setup(rank:int,world_size:int):
    """
    Function for setting up DDP enviroinment
    
    :param rank: (int) current device index
    :param world_size: (int) Total number of devices
    """
    os.environ['MASTER_ADDR']='127.0.0.8'
    os.environ['MASTER_PORT']='17929'
    init_process_group(backend='nccl',rank=rank,world_size=world_size)
