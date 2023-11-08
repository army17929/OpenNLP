"""
Description: Data Labeling module for AIMS NLP project
Author: O Hwang Kwon 
Version: 0.0.1
Python Version: 3.10.11
Dependencies: stanza, textblob, vaderSentiment, pattern, tweetnlp, pysentiment2
License: n/a
"""

import argparse
import pandas as pd
import statistics
from pattern.en import sentiment
import stanza
import tweetnlp
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweetnlp
from transformers import pipeline
import pysentiment2 as ps

class label():
    """
    Data Labeling module 

    Data labeling is the largest bottleneck for supervised learning. 
    This module provides text sentiment labeling, using seven open source lexicon based libraries.
    Textblob,vader,stanza,pattern,tweetnlp,TwitRoBERT, pysentiLM are those. 
    We intentionally selected seven(odd number) libraries, because the final label will be determined with majority voting system.

    :param fileName: (str) input file name ; expect to be ``.csv`` file.
    :param output_file: (str) output file name ; expect to be ``.csv`` file
    :param tools: (str) specify the tools for sentiment analysis. Default=``all``
    """
    def __init__(self,fileName:str):
        self.Tool_list=['TextBlob','VADER','Stanza','pattern','TweetNLP','TwitRoBERT','pysentiLM']
        self.default='TextBlob VADER Stanza pattern TweetNLP TwitRoBERT pysentiLM'
        self.fileName=fileName
        self.listOfScores=[]
        self.listOfTools=self.Tool_list
        self.df=pd.read_csv(self.fileName,encoding='latin1')
        self.df['tweets']=self.df['tweets'].astype(str)

        # Print out the file name on the terminal 
        print(f"Input file: {fileName}")

    def getScorefromTXT(self,txtlabel):
        """
        Convert sentiment label(str) to integer(int)
        """
        if str(txtlabel).lower()=='negative':
            return -1 
        elif str(txtlabel).lower()=='neutral':
            return 0
        elif str(txtlabel).lower()=='positive':
            return 1 

    def getPatternScore(self,text):
        return  sentiment(text)[0]

    def normalize(self,newRangeList, oldRangeList, value):
        numer=float((value-oldRangeList[1])*(newRangeList[0]-newRangeList[1]))
        denom=float(oldRangeList[0]-oldRangeList[1])
        return float(newRangeList[1])+float((numer)/(denom))

    def getFinalLabel(self,score):
        """
        Convert from score(float) to sentiment(str)
        """
        theScore=float(score)
        if (theScore<-0.05) & (theScore>=-1):
            return "Negative"
        elif (theScore>0.05) & (theScore<=1):
            return "Positive"
        elif (theScore>=-0.5) & (theScore<=0.05):
            return "Neutral"
        else:
            return "OUT OF BOUNDS"

    def setup(self):
        """
        Library setup function 

        Since labeling tools have different dependencies, it is better to check whether each tool is set up properly.
        """
        #Stanza 
        import stanza
        self.df['Stanza']='' # Create a new column
        print("Successfully set-up Stanza")
            
        #TextBlob 
        from textblob import TextBlob
        self.df['TextBlob']=''
        print("Successfully set-up TextBlob")


        #VADER 
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.df['VADER']=''
        print("Successfully set-up VADER")

        #pattern
        from pattern.en import sentiment
        self.df['pattern']=''
        print("Successfully set-up pattern")


        #TweetNLP 
        import tweetnlp
        self.df['TweetNLP']=''
        print("Successfully set-up TweetNLP")


        #TwitRoBERT
        from transformers import pipeline
        self.df['TwitRoBERT']=''
        print("Successfully set-up TwitRoBERT")

        #pysentiLM
        import pysentiment2 as ps
        self.df['pysentiLM']=''
        print("Successfully set-up pysentiLM")

        #CREATE stats COLUMNS
        self.df['mean']=''
        self.df['StandardDev']='' # Create std column
        self.df['FinalScore']='' # Final score column
        print('SETUP COMPLETE')

    def label(self,output_file_average:str,
              output_file_vote:str):
        """
        Comprehensive labeling function 

        This function labels the text using selected libraries. 
        Each tool will output sentiment polarity score. 
        Those scores will be provided as a float bewteen ``[-1,1]``. 
        At the end, statistics of those polarity scores(mean,std) will be provided as well.
        The result will be saved as a ``.csv`` file.

        :input output_file: (str) output file name, expect to be csv.file
        """

        # Create instances ahead of for-loop
        nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_no_ssplit=True)
        vaderPre=SentimentIntensityAnalyzer()
        model = tweetnlp.Sentiment() # Create an instance
        sentiment_taskTWR = pipeline("sentiment-analysis", 
                        model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                        tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest')
        lm = ps.LM()

        for indx in self.df.index:
            tempSum=0
            text=self.df['tweets'].iloc[indx]
            
            #STANZA
            def get_sentiment_stanza(text):
                """
                Function for getting sentiment from stanza

                :Input text: (str) text for sentiment analysis
                """
                textS=nlp(text)
                for i, sentence in enumerate(textS.sentences):
                    if sentence.sentiment==0:
                        return -1 # Negative
                    elif sentence.sentiment==1:
                        return 0 # Neutral
                    elif sentence.sentiment==2:
                        return 1 # Positive
            stanzaVal=get_sentiment_stanza(text)
            self.df.at[indx,'Stanza']= stanzaVal # Save at the dataframe
            tempSum=tempSum+stanzaVal
            self.listOfScores.append(stanzaVal)

            # TEXTBLOB
            textBlobVal=TextBlob(text).sentiment.polarity
            self.df.at[indx,'TextBlob']= textBlobVal
            tempSum=tempSum+textBlobVal
            self.listOfScores.append(textBlobVal)

            # VADER
            vaderVal=vaderPre.polarity_scores(text)['compound']
            self.df.at[indx,'VADER']= vaderVal
            tempSum=tempSum+vaderVal
            self.listOfScores.append(vaderVal)

            # PATTERN
            patternVal=sentiment(text)[0]
            self.df.at[indx,'pattern']= patternVal
            tempSum=tempSum+patternVal
            self.listOfScores.append(patternVal)

            # TWEETNLP
            labelVar=model.sentiment(text)['label']
            TweetNLPVal=self.getScorefromTXT(labelVar)
            self.df.at[indx,'TweetNLP']= TweetNLPVal
            tempSum=TweetNLPVal+tempSum
            self.listOfScores.append(TweetNLPVal)

            # TWITROBERT
            twitRoBERT=sentiment_taskTWR(text)[0]['label']
            twitRoBERTVal=self.getScorefromTXT(twitRoBERT)
            self.df.at[indx,'TwitRoBERT']= twitRoBERTVal
            tempSum=tempSum+twitRoBERTVal
            self.listOfScores.append(twitRoBERTVal)

            # PYSENTI LM
            tokensl = lm.tokenize(text)
            pysentiLMVAL=lm.get_score(tokensl)['Polarity']
            self.df.at[indx,'pysentiLM']= pysentiLMVAL
            tempSum=tempSum+pysentiLMVAL
            self.listOfScores.append(pysentiLMVAL)
            
            # Average score 
            tempAverage=tempSum/len(self.listOfTools)

            if indx%10==1:
                print("On row ", indx," ") # Every 10 rows, program will notify the progress.
            
            #store average
            self.df.at[indx,'mean']=tempAverage
            self.df.at[indx,'StandardDev']=statistics.stdev(self.listOfScores)
            self.df.at[indx,'FinalLabel']= self.getFinalLabel(tempAverage)
            
            #reset variables
            self.listOfScores.clear()
            tempSum=0
        # Once this for-loop is done, the dataframe will have each score from each tool.
        # Stats from those scores will be included as well.
        df_average=self.df
        df_average.to_csv(output_file_average,mode='w')
    
        def majority_voting(df=df_average):
            """
            Final label generating function 

            This function could be used only when the user used all 7 tools.
            Among 7 tools, only 4 libraries offer polarity score.

            This function will take result from ``label()`` into account using majority voting system.
            If majority voting does not work, it will label using sentiment polarity score. 
            Example. If 3 tools said 'Positive', 3 tools said 'Neutral', and 1 tool said 'Negative', Final label will be determined by mean polarity score.
            if ``mean score<=-0.05 : Negative``
            elif ``mean score>=0.05 : Positive``
            else ``Neutral`` 
            """

            # Preparation for the majority voting system.
            def transform_value(value):
                if value>=0.05 :
                    return 1 
                elif value<=-0.05:
                    return -1
                else :
                    return 0
                
            # For the libraries that provide polarity score, 
            # Convert the score into labels.
            columns_to_transform = ['TextBlob','pattern','VADER','pysentiLM']
            for column in columns_to_transform :
                df[column] = df[column].apply(transform_value)


            #Build new columns in the dataframe.
            #Note: This will build new columns that indicate the majority vote.
            columns_to_count=['Stanza','TextBlob','VADER','pattern','TweetNLP','TwitRoBERT','pysentiLM']
            L=[]
            for i in range (7):
                index=df.columns.get_loc(columns_to_count[i])
                L.extend([index])
                print(f"index of column {columns_to_count[i]} is {index}")
            
            df['count_positive'] = (df.iloc[:, min(L):max(L)+1] == 1).sum(axis=1)
            df['count_neutral'] = (df.iloc[:, min(L):max(L)+1] == 0).sum(axis=1)
            df['count_negative'] = (df.iloc[:, min(L):max(L)+1] == -1).sum(axis=1)
            print('Columns counted : ',df.columns[min(L):max(L)+1])
            
            columns_to_compare = ['count_positive','count_neutral','count_negative']

            # Note: When there is no need to consider the average.
            def vote(df=df):
                # IF there are two competing labels when labeled by voting system,
                if (df['count_positive'] == 3 and df['count_neutral'] == 3) or \
                (df['count_negative'] == 3 and df['count_positive'] == 3) or \
                (df['count_negative'] == 3 and df['count_neutral'] == 3):
                    if df['mean'] > 0.05: 
                        return "Positive"
                    elif df['mean'] < -0.05:
                        return "Negative"
                    else:
                        return "Neutral"
                else :
                    if max(df[columns_to_compare])==df['count_positive']:
                        return"Positive"
                    elif max(df[columns_to_compare])==df['count_neutral']:
                        return"Neutral"
                    else:
                        return'Negative'

            df['FinalLabel'] = df.apply(vote, axis=1)
            #df=df.drop('FinalScore',axis=1)
            df.to_csv(f'{output_file_vote}',mode='w')
            print(f"Data labeling by voting is completed. Output file : {output_file_vote}")
        
        majority_voting(df=df_average)
        print(f"Data labeling by averaging is completed.")

if __name__=="__main__":
    LABEL=label(fileName='sample_data.csv')
    LABEL.setup()
    LABEL.label(output_file_average='average.csv',
                output_file_vote='vote.csv')
    
    