from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer,BitsAndBytesConfig

class CustomClassificationModel(torch.nn.Module):
    """
    Custome Classification Model module 
    
    :param checkpoint: (str) checkpoint of the model you want to use. Example: ``bert-base-uncased``
    :param num_class: (int) number of class in classification problem. 
    """
    def __init__(self,checkpoint,num_class):
        super(CustomClassificationModel,self).__init__()
        self.checkpoint=checkpoint
        self.model=AutoModel.from_pretrained(checkpoint)
        self.fc1=torch.nn.Linear(self.model.config.hidden_size,16)
        self.fc2=torch.nn.Linear(16,num_class)
        self.dropout=torch.nn.Dropout(0.1)

    def forward(self,input,mask):
        #"""
        #Forward Propagation function 
        #- Input
        #    input : input_id tensor
        #    mask : attention_mask tensor 
        #- Return 
        #    Logit
        #"""
        outputs=self.model(input_ids=input,attention_mask=mask) # output from the pretrained model 
        last_hidden_state=outputs.last_hidden_state
        pooled_output=torch.mean(last_hidden_state,dim=1)
        pooled_output=pooled_output.to(torch.float32)
        x=F.relu(self.fc1(pooled_output))
        output=F.softmax(self.fc2(x),dim=1)
        x=self.dropout(x)
        return output
    
class PEFTClassificationModel(torch.nn.Module):
    """
    Custom Classification Model for PEFT(Parameter Efficient Fine Tuning) module
    
    :param model: (object) quantized model for the problem
    :param num_class: (int) number of class in classification problem. 
    """
    def __init__(self,model,num_class):
        super(PEFTClassificationModel,self).__init__()
        self.model=model
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

class peft():
    """
    Parameter Efficient Fine Tuning(PEFT) Preparation module

    :param checkpoint: (str) model's checkpoint. Example ``bert-base-uncased``
    """
    def __init__(self,checkpoint:str,load_in_4bit=True):
        def create_quantization_config():
            if load_in_4bit:    
                bnb_config=BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='fp4',
                    bnb_4bit_compute_dtype=torch.bfloat16)
            return bnb_config
        self.checkpoint=checkpoint
        self.model=AutoModel.from_pretrained(checkpoint,
                                quantization_config=create_quantization_config())
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))

    def find_all_linear_names(self):
        #"""
        #This function will return a list of layers that Low Rank Adaptation(LoRA) is applicable. 
        #"""
        cls=torch.nn.Linear
        lora_module_names=set() 
        for name,module in self.model.named_modules():
            if isinstance(module,cls):
                names=name.split('.')
                lora_module_names.add(names[0] if len(names)==1 else names[-1])
            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_peft_config(self,r=512,lora_alpha=32,lora_dropout=0.0):
        """
        Low Rank Adaptation(LoRA) config generator

        :param r: (int) The rank of update matrices. Lower rank results in smaller update matrices with fewer trainable parameters.
        :param lora_alpha: (int) LoRA scaling factor 
        :param dropout: (float) Dropout probability during training
        """
        peft_config=LoraConfig(r=r,lora_alpha=lora_alpha,
                            target_modules=self.find_all_linear_names(),
                            lora_dropout=lora_dropout,
                            bias='none')
        return peft_config

    def train_preparation(self):
        #"""
        #Comprehensive preparation function for peft
        #"""
        # Prepare the model for the kbit training
        model=prepare_model_for_kbit_training(self.model)

        # Find out the modules and layers that lora can be applied.
        target_modules=self.find_all_linear_names()
        peft_config=self.create_peft_config(target_modules) # Create configuration
        model=get_peft_model(model,peft_config=peft_config) # Model modification done.
        print('MODEL SUMMARY \n',model)
        print_trainable_parameters(model)
        return model

def print_trainable_parameters(model,use_4bit=False):
    #"""
    #This is a helper function that you can compute the percentage of parameters that could be trained with LoRA.
    #The more you compress the model, the less you will have trainable parameters. 
    #"""
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
