""" 
            Ex 5. Fine Tuning Large Language Models using OpenNLP(FSDP)

When your computing source cannot afford heavy langauge model with
a few billioin parameters, FSDP would be a good solution for that.
OpenNLP supports FSDP, and it significantly reduces the engineering 
complexity of model parallelism.
"""

from opennlp.run.LLM import BERT,GPT,Llama

if __name__=="__main__":
    """
    Note that you can only implement FSDP using ``torchrun``.
    With torchrun you can handle multi machine computing environment. 
    Ex) ``torchrun --nnodes 1 --nproc_per_node gpu ex5_LLM_FSDP.py``
    """
    # Constants
    data_path='./data/sample_sentiment.csv'
    input_col='tweets'
    output_col='labels'
    epochs=3
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

    bert.run_BERT_FSDP(world_size=world_size,
                    epochs=epochs,
                bs=bs,
                lr=lr,
                save_every=1)

    gpt.run_GPT_FSDP(world_size=world_size,
                epochs=epochs,
                bs=bs,
                lr=lr,
                save_every=1)
