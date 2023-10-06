from pathlib import Path
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch import Tensor
from typing import Iterator,Tuple
import torchmetrics
import numpy as np
import torch
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import time
import bitsandbytes as bnb
import os
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

# Let us define a function that plots training and validation plot 
def PlotTraining(history,savedir:str,model_name:str):
    save_dir='./Results'+save_dir
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
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(f"{savedir}/Trainingplot_{model_name}")

def prepare_const(num_epochs:int,batch_size:int,
                  lr:float,save_every:int,model_name:str)->dict:
    # Data and model directory + Training hyperparameters
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

# Each process control a single GPU
def ddp_setup(rank:int,world_size:int):
    os.environ['MASTER_ADDR']='127.0.0.8'
    os.environ['MASTER_PORT']='20502'
    init_process_group(backend='nccl',rank=rank,world_size=world_size)


def metrics_generator(y_true:list,y_pred:list,save_dir:str,model_name:str):
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
        os.makedirs(save_dir)
    save_path=os.path.join(save_dir,f"cm_{model_name}")
    plt.savefig(save_path)
    # Create the classification report 
    report=classification_report(y_true,y_pred)
    file=f"/{model_name}_classification_report.txt"
    with open(save_dir+file,'w') as f :
        f.write(report)

def dataloader_single(trainset:Dataset,testset:Dataset,valset:Dataset,bs:int)->Tuple[DataLoader,DataLoader,DataLoader]:
    trainloader=DataLoader(trainset,batch_size=bs,shuffle=True,num_workers=1)
    testloader=DataLoader(testset,batch_size=bs,shuffle=False,num_workers=1)
    valloader=DataLoader(valset,batch_size=bs,shuffle=False,num_workers=1)
    return trainloader,testloader,valloader

def dataloader_ddp(trainset:Dataset,testset:Dataset,
                   valset:Dataset,bs:int)-> Tuple[DataLoader,DataLoader,DataLoader,DistributedSampler,DistributedSampler]:
    sampler_train=DistributedSampler(trainset)
    sampler_val=DistributedSampler(valset)
    # Note that shuffle=False
    trainloader=DataLoader(trainset,batch_size=bs,
                           shuffle=False,sampler=sampler_train,
                           num_workers=2)
    testloader=DataLoader(testset,batch_size=bs,shuffle=False,
                          sampler=sampler_val,
                          num_workers=2)
    valloader=DataLoader(valset,batch_size=bs,
                           shuffle=False,sampler=DistributedSampler(valset,shuffle=False),
                           num_workers=2)
    return trainloader,testloader,valloader,sampler_train,sampler_val

class TrainerSingle:
    def __init__(self,gpu_id:int,model:nn.Module,
                 trainloader:DataLoader,
                 testloader:DataLoader,
                 valloader:DataLoader,const):
        self.gpu_id=gpu_id # Current working thread
        self.const=const # Learning parameters 
        self.model=model.to(self.gpu_id) # Send the model to GPU
        self.trainloader=trainloader 
        self.testloader=testloader
        self.valloader=valloader
        self.criterion=nn.CrossEntropyLoss()
        self.optimizer=bnb.optim.AdamW8bit(self.model.parameters(),
                                 lr=self.const['lr'])
        self.lr_scheduler=optim.lr_scheduler.StepLR(
            self.optimizer,self.const['lr_step_size']
        )

        self.train_acc=torchmetrics.Accuracy(
            task="multiclass",num_classes=3,average="micro"
        ).to(self.gpu_id)

        self.val_acc=torchmetrics.Accuracy(
        task="multiclass",num_classes=3,average="micro"
        ).to(self.gpu_id)

        self.test_acc=torchmetrics.Accuracy(
            task="multiclass",num_classes=3,average="micro"
        ).to(self.gpu_id)
        self.train_acc_array=torch.zeros(self.const['total_epochs'])
        self.val_acc_array=torch.zeros(self.const['total_epochs'])
        self.train_loss_array=torch.zeros(self.const['total_epochs'])
        self.val_loss_array=torch.zeros(self.const['total_epochs'])
        
    def _run_batch(self,src:list,tgt:Tensor)->float:
        self.optimizer.zero_grad() 
        out=self.model(src[0],src[1])
        loss=self.criterion(out,tgt)
        loss.backward()
        self.optimizer.step()

        self.train_acc.update(out,tgt)
        self.val_acc.update(out,tgt)
        return loss.item()
    
    def _run_epoch(self,epoch:int):
        loss=0.0
        val_loss=0.0
        # Run on the training set
        self.train_acc.reset()
        for input,mask,tgt in self.trainloader:
            input=input.to(self.gpu_id)
            mask=mask.to(self.gpu_id)
            tgt=tgt.to(self.gpu_id)
            src=[input,mask] # LLM need both input_ids and attention_masks
            loss_batch=self._run_batch(src,tgt)
            loss+=loss_batch
        self.lr_scheduler.step()
        self.train_acc_array[epoch]=self.train_acc.compute().item()
        self.train_loss_array[epoch]=loss/len(self.trainloader)
        
        # Run on the validation set
        self.val_acc.reset()
        for input,mask,tgt in self.valloader:
            input=input.to(self.gpu_id)
            mask=mask.to(self.gpu_id)
            tgt=tgt.to(self.gpu_id)
            src=[input,mask] # LLMs need two inputs. 
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
        
        if epoch==self.const['total_epochs']-1 : 
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
            save_dir=f"./Results/{self.const['model_name']}_epoch_{self.const['total_epochs']}_batch_{self.const['batch_size']}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/Trainingplot_{self.const['model_name']}_epoch_{self.const['total_epochs']}_bs_{self.const['batch_size']}.png")

    def _save_checkpoint(self,epoch:int):
        checkpoint=self.model.state_dict()
        model_path=self.const["trained_models"]/f"Nuclear_epoch{epoch}.pt"
        torch.save(checkpoint,model_path)
    
    def train(self,max_epochs:int):
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.const['save_every']==0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(max_epochs-1) # save the last epoch
    
    def test(self,final_model_path:str):
        self.model.load_state_dict(torch.load(final_model_path))
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
        metrics_generator(y_true=y_true,y_pred=y_pred,
                     save_dir=f"/{self.const['model_name']}_epoch_{self.const['total_epochs']}_batch_{self.const['batch_size']}",
                     model_name=f"{self.const['model_name']}")

# This is a trainer class for Multi GPUs training
class TrainerDDP(TrainerSingle):
    def __init__(
            self, gpu_id:int, model:nn.Module,
            trainloader:DataLoader,testloader:DataLoader,
            valloader:DataLoader, 
            sampler_train:DistributedSampler,
            sampler_val:DistributedSampler,
            const
    )->None:
        super().__init__(gpu_id,model,trainloader,
                         testloader,valloader,const)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        # Wrap the model with DDP
        self.model=DDP(self.model,device_ids=[gpu_id],find_unused_parameters=True)
        self.sampler_train=sampler_train
        self.sampler_val=sampler_val
    
    def _save_checkpoint(self, epoch: int):
        checkpoint=self.model.module.state_dict()
        model_path=self.const["trained_models"]/f"Nuclear_epoch{epoch}.pt"
        torch.save(checkpoint,model_path)
    
    def train(self,max_epochs:int):
        self.model.train()
        for epoch in range(max_epochs):
            self.sampler_train.set_epoch(epoch)
            self.sampler_val.set_epoch(epoch)
            self._run_epoch(epoch)
            if epoch%self.const["save_every"]==0:
                self._save_checkpoint(epoch)
        # Save last epoch 
        self._save_checkpoint(max_epochs-1)
    
    def test(self,final_model_path:str):
        print(final_model_path)
        self.model.module.load_state_dict(
            torch.load(final_model_path,map_location="cpu"),
            #torch.load('/home/ohwang/LLM_Engine/trained_BERT/Nuclear_epoch4.pt',map_location="cpu"),
            )
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
        metrics_generator(y_true=y_true,y_pred=y_pred,
                     save_dir=f"/{self.const['model_name']}_epoch_{self.const['total_epochs']}_batch_{self.const['batch_size']}",
                     model_name=f"{self.const['model_name']}")

# Finally, this is the main function that trains the model 
def main_single(gpu_id:int,final_model_path:str,
                model:nn.Module,
                train_dataset:Dataset,
                test_dataset:Dataset):
    const=prepare_const()
    train_dataloader,test_dataloader=dataloader_single(
        train_dataset,test_dataset,const["batch_size"]
    )
    trainer=TrainerSingle(
        gpu_id=gpu_id,model=model,
        trainloader=train_dataloader,
        testloader=test_dataloader)
    trainer.train(const["total_epochs"])
    trainer.test(final_model_path)

def main_ddp(rank:int,world_size:int,
             model:nn.Module,
             train_dataloader:DataLoader,
             test_dataloader:DataLoader,
             val_dataloader:DataLoader,
             train_sampler:DistributedSampler,
             val_sampler:DistributedSampler,
             final_model_path:str):
    # Initialize the environment
    ddp_setup(rank,world_size)

    const=prepare_const()

    trainerDDP=TrainerDDP(
        gpu_id=rank,model=model,trainloader=train_dataloader,
        valloader=val_dataloader,
        testloader=test_dataloader,
        sampler_train=train_sampler,
        sampler_val=val_sampler,const=const)
    
    trainerDDP.train(const["total_epochs"])
    trainerDDP.test(final_model_path)

    destroy_process_group() # Clean up 
