"""
    Ex 5. Performing Data Labelling using OpenNLP

    OpenNLP provides 5 sentiment analysis tools-
    1.VADER
    2.pattern
    3.TweetNLP
    4.TwitRoBERT
    5.pysentiLM

    Different properties like mean, majority result and Standard Deviation of the tweets can be calculated using OpenNLP. 

"""


from opennlp.labeling.labeling import label

#Create instance of Label class 
#fileName argument is used to take in the input file containing tweets to be labelled
LABEL=label(fileName='sample_nuclear.csv')

#To instantiate different sentiment analysis tools
LABEL.setup()
    
#To label tweets
#Output_file_average argument saves average sentiment score each tweet through different libraries in csv file.
#output_file_vote argument saves the majority voting result of different sentiment analysis libraries in csv file.   
LABEL.label(output_file_average='average.csv',
                output_file_vote='vote.csv')
    