import stanza
import argparse
import pandas as pd
import statistics
import gc
from textblob import TextBlob 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

toolOpt=[
    'TextBlob',
    'VADER',
    'Stanza', 
    'pattern',
    'TweetNLP',
    'TwitRoBERT',
    'pysentiLM',
    'all'
    ]

allToolsList=[
    'TextBlob',
    'VADER',
    'Stanza', 
    'pattern',
    'TweetNLP',
    'TwitRoBERT',
    'pysentiLM',
]
defaultTool= 'TextBlob VADER Stanza pattern TweetNLP TwitRoBERT pysentiLM'

listOfTools=allToolsList

c=0
class Labelling:

    def __init__(self,ltools,df,sflag):

        self.tools=ltools
        self.df=df
        self.sdflag=sflag
    
    @staticmethod
    def transform_value(value):
        if (value>=0.05):
            return 1 
        elif (value<=-0.05):
            return -1
        else :
            return 0
    
    @staticmethod
    def get_finalScore(score):
        theScore=float(score)
        if (theScore<-0.05) & (theScore>=-1):
            return "Negative"
        elif (theScore>0.05) & (theScore<=1):
            return "Positive"
        elif (theScore>=-0.5) & (theScore<=0.05):
            return "Neutral"
        else:
            return "OUT OF BOUNDS"

    @staticmethod
    def getScorefromTXT(txtlabel):
        if str(txtlabel).lower()=='negative':
            return -1
        elif str(txtlabel).lower()=='neutral':
            return 0
        elif str(txtlabel).lower()=='positive':
            return 1
        else:
            return -1000000000000000000000# means error 
        
    @staticmethod
    def getPatternScore(text):
        return sentiment(text)[0]

    def Comp_Stanza(self):

        import stanza
        nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
        c=0
        def stanza_sentiment(text):

            global c
            c+=1
            print(c)
            doc = nlp(text)
            for i, sentence in enumerate(doc.sentences):
                if sentence.sentiment==0:
                        return -1
                elif sentence.sentiment==1:
                        return 0
                elif sentence.sentiment==2:
                        return 1 


        self.df['Stanza'] = self.df['tweets'].apply(stanza_sentiment)

        del nlp
        
        gc.collect()

    def Comp_TextBlob(self):

        #TextBlob
        self.df['TextBlob'] = self.df['tweets'].apply(lambda i:TextBlob(i).sentiment.polarity)

    def Comp_VADER(self):
         
        vaderPre=SentimentIntensityAnalyzer()

        self.df['VADER'] = self.df['tweets'].apply(lambda i:vaderPre.polarity_scores(i)['compound'])

        del vaderPre
        
        gc.collect()
    
    def Comp_Pattern(self):
         
        import pattern
        from pattern.en import sentiment
        

        self.df['pattern'] = self.df['tweets'].apply(lambda i:sentiment(i)[0])

    def Comp_tweetnlp(self):
         

        import tweetnlp
        model = tweetnlp.Sentiment() 

        self.df['TNLP_eng'] = self.df['tweets'].apply(lambda i:model.sentiment(i)['label'])

        self.df['TweetNLP']= self.df['TNLP_eng'].apply(lambda i:Labelling.getScorefromTXT(i))

        del model
        
        gc.collect()
    
    def Comp_TwitRoberta(self):
         
        from transformers import pipeline
    
        sentiment_taskTWR = pipeline("sentiment-analysis", model='cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest')

        self.df['TwitRoBERT_eng'] = self.df['tweets'].apply(lambda i:sentiment_taskTWR(i)[0]['label'])

        self.df['TwitRoBERT']=self.df['TwitRoBERT_eng'].apply(lambda i:Labelling.getScorefromTXT(i))

        del sentiment_taskTWR
        
        gc.collect()

    def Comp_pysenti(self):
         
        import pysentiment2 as ps
    
        lm = ps.LM()

        def pysentiscore(text):

            tokensl = lm.tokenize(text)
            pysentiLMVAL=lm.get_score(tokensl)['Polarity']
            return pysentiLMVAL

        self.df['pysentiLM']=self.df['tweets'].apply(lambda i:pysentiscore(i))

        del lm
        
        gc.collect()

    def Comp_mean(self):
             
        self.df['mean'] = df[self.tools].mean(axis=1)

    def Comp_SDev(self):
         
        self.df['Std_Dev']=df[self.tools].std(axis=1)

    
    def Comp_Sentiment(self):
         
        if 'Stanza' in self.tools:
             
            self.Comp_Stanza()

            #print("Stanza done")

            
        if 'TextBlob' in listOfTools:

            self.Comp_TextBlob()
            #print("Blob done")
            
            
        if 'VADER' in listOfTools:

            self.Comp_VADER()
            #print("Vader done")
            

        if 'pattern' in listOfTools:

            self.Comp_Pattern()
            #print("Pattern done")
            

        if 'TweetNLP' in listOfTools:

            self.Comp_tweetnlp()
            #print("TwNLP done")

        if 'TwitRoBERT' in listOfTools:

            self.Comp_TwitRoberta()
            #print("Robert done")

        if 'pysentiLM' in listOfTools:

            self.Comp_pysenti()
            #print("Senti done")
            

        self.Comp_mean()
        
        self.Comp_SDev()

        df[self.tools]=df[self.tools].round(2)

        

        columns_to_transform = ['TextBlob','pattern','VADER','pysentiLM']

        for column in columns_to_transform :
            self.df[column] = df[column].apply(lambda i: Labelling.transform_value(i))
        

        df['count_positive'] = (df.iloc[:, 5:12] == 1).sum(axis=1)
        df['count_neutral'] = (df.iloc[:, 5:12] == 0).sum(axis=1)
        df['count_negative'] = (df.iloc[:, 5:12] == -1).sum(axis=1)
        
        columns_to_compare = ['count_positive','count_neutral','count_negative']

        self.df['FinalScore_mine']=self.df['mean'].apply(lambda i: Labelling.get_finalScore(i))
        
        self.df['Final_label'] = df.apply(lambda i:self.Majority_voting(i,columns_to_compare), axis=1)

        return self.df


        
    def Majority_voting(self,i,columns_to_compare):
        if (i['count_positive'] == 3 and i['count_neutral'] == 3) or (i[ 'count_negative'] == 3 and i['count_positive'] == 3) or ( i['count_negative'] == 3 and i['count_neutral'] == 3):
            if i['mean'] > 0.05: 
                return "Positive"
            elif i['mean'] < -0.05:
                return "Negative"
            else:
                return "Neutral"
        else :
            if max(i[columns_to_compare])== i['count_positive']:
                return"Positive"
            elif max(i[columns_to_compare])== i['count_neutral']:
                return"Neutral"
            else:
                return'Negative'


#Get arguments from commandline

#Gets arguments!
parser = argparse.ArgumentParser(description="inputs dataset file, tools; outputs average sentiment values;assumes column named tweets") #creates parser
parser.add_argument('fileName',help='input of file name; expects a CSV')
parser.add_argument('-tools',nargs='*',choices=toolOpt,default=defaultTool, help='list of tools to run; \'all\' option runs all tools')#requires at least 1 tool

#file name 
args =parser.parse_args()

fileName=args.fileName  
fileNameOut='Data_labeling_completed.csv'

#Study specific/sentiment polarity stuff
listRange=[-1,1]

df = pd.read_csv(fileName, encoding='latin1')

df=df[['tweets']]

df['tweets'] = df['tweets'].astype(str) 

df=df[:10]

if 'all' in args.tools:
    listOfTools=allToolsList
    print
else:
    listOfTools=args.tools

listOfScores=[]#scores for each tool 
numberOfToolsEP=len(args.tools)

#See if can calculate Standard Deviation 
if numberOfToolsEP>=2:
    SDevFlag=True
elif 'all' in args.tools: 
    SDevFlag=True
else:
    SDevFlag=False

listOfTools=allToolsList

l=Labelling(listOfTools,df,1)

labelled_data=l.Comp_Sentiment()

labelled_data.to_csv("Data_Labelling_0710")
