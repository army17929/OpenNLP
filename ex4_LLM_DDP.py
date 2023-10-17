""" 
            Ex 3. Fine Tuning Large Language Models using OpenNLP(Multiple GPUs)

OpenNLP provides handy fine tuning tools for 3 LLMs, also for Distributed learning.  
Currently we do not support distributed learning function for Llama, 
because quantized model is not supported by Pytorch DDP library.
"""

from opennlp.run.LLM import BERT,GPT,Llama

if __name__=="__main__":
    """
    Note that you must specify this if statement, otherwise you will 
    encounter Childprocess error.

    The reason for the error is that if you execute this program, 
    same process will run through multiple devices(GPUs). Therefore,
    without ``if __name__=="__main__":``, the main process will crash 
    with child process.
    """
    # Constants
    data_path='./data/sample_sentiment.csv'
    input_col='tweets'
    output_col='labels'
    epochs=1
    bs=32
    lr=1e-5
    world_size=2 # Assume that we are training the models with 2 GPUs

    """
    You don't have to change the instances, which makes DDP very handy!
    """
    bert=BERT(data_path=data_path,
            input_col=input_col, 
            output_col=output_col,
            num_class=2)

    gpt=GPT(data_path=data_path,
            input_col=input_col, 
            output_col=output_col,
            num_class=2)

    bert.run_BERT_DDP(world_size=world_size,
                    epochs=epochs,
                bs=bs,
                lr=lr,
                save_every=1)

    gpt.run_GPT_DDP(world_size=world_size,
                epochs=epochs,
                bs=bs,
                lr=lr,
                save_every=1)
