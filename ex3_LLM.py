""" 
            Ex 3. Fine Tuning Large Language Models using OpenNLP(Single GPU)

OpenNLP provides handy fine tuning tools for 3 LLMs.  
1. BERT - Lightest
2. GPT - Moderate
3. Llama - Massive
Runtime of the model will be proportional to the number of parameters.
"""

from opennlp.run.LLM import BERT,GPT,Llama

# Constants
data_path='./data/sample_sentiment.csv'
input_col='tweets'
output_col='labels'
epochs=1
bs=32
lr=1e-5

# Create instances from the classes.
bert=BERT(data_path=data_path,
          input_col=input_col, 
          output_col=output_col,
          num_class=2)

gpt=GPT(data_path=data_path,
          input_col=input_col, 
          output_col=output_col,
          num_class=2)

llama=Llama(data_path=data_path,
          input_col=input_col, 
          output_col=output_col,
          num_class=2)

bert.run_BERT(epochs=epochs,
              bs=bs,
              lr=lr,
              save_every=1)

gpt.run_GPT(epochs=epochs,
              bs=bs,
              lr=lr,
              save_every=1)

llama.run_LLAMA(epochs=epochs,
              bs=8,
              lr=lr,
              save_every=1)