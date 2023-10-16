import requests # This library is for the request to HTTP
import os # Interacting with operating system
import json
import configparser
import csv
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import shutil

class Data_Mining():
    """
    Data Mining Module 
    """
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(r'config.ini')
        self.api_key = config['twitter']['api_key']
        self.api_key_secret = config['twitter']['api_key_secret']
        self.access_token = config['twitter']['access_token']
        self.access_token_secret = config['twitter']['access_token_secret']
        self.bearer_token = config['twitter']['bearer_token']
        self.search_url="https://api.twitter.com/2/tweets/search/recent"

    def query(self,max_result:int,keyword:str,must_include:str):
        """
        Query Generation function 

        :arg max_result: (int) number of maximum tweets per a single time. This depends on API tier you are using.
        :arg keyword: (str) keyword thay you want to include in query
        :arg must_include: (str) keyword that must be included in the response text.
        """
        query = f'"{must_include}" {keyword} lang:en -is:retweet'
        query_params = {
            'query': query,
            'max_results' : max_result,
            'expansions' : 'attachments.poll_ids,attachments.media_keys,author_id,geo.place_id,in_reply_to_user_id,referenced_tweets.id,entities.mentions.username,referenced_tweets.id.author_id,edit_history_tweet_ids',
            'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,edit_controls,edit_history_tweet_ids,entities,geo,id,in_reply_to_user_id,lang,note_tweet,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text,withheld'
        }
        return query_params

    def bearer_oauth(self,r):
        #"""
        #Method required by bearer token authentication.
        #"""
        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2RecentSearchPython"
        return r
    
    def connect_to_endpoint(self,url,params):
        response = requests.get(url, auth=self.bearer_oauth, params=params)
        print(response.status_code)
        if response.status_code != 200: # 200 means that the request is successful.
            raise Exception(response.status_code, response.text)
        return response.json()
    
    def json_to_csv(json_file, csv_file):
        with open(json_file, 'r') as file:
            tweets = json.load(file)

        with open(csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # If this is the first row,
            if file.tell() == 0 :
                # Head of the csv.file
                writer.writerow(['text','Keywords','Author ID','Created At'])

            for tweet in tweets['data']:
                text = tweet['text']
                author_id = tweet['author_id']
                created_at = tweet['created_at']

                writer.writerow([text, author_id, created_at])
        print("CSV file saved successfully.")

if __name__=="__main__":
    dm=Data_Mining()
    qp=dm.query()
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    json_response = dm.connect_to_endpoint(search_url, qp)
        
    # Files to be stored
    json_file ='tweets.json'
    csv_file = 'Tweets.csv' # csv.file to be stored
    output_directory = r'./data'
        
    # Building a JSON file usin JSON response.
    with open(json_file, 'w') as file :
        json.dump(json_response, file)

    # Converting json to csv file.
    csv_path = os.path.join(output_directory, csv_file)
    dm.json_to_csv('tweets.json',csv_path)
    print("csv file is saved successfully as \"Tweets.csv\". ")
    print(f"Saved directory : {csv_path}")