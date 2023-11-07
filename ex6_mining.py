"""
    Ex 6. Performing Data Mining using OpenNLP

    OpenNLP provides functionality to mine data from twitter using its search endpoints.

"""
from opennlp.mining.api import Data_Mining
import json
import os

#Create instance from the class
dm=Data_Mining()

#Generate Query
qp=dm.query(100,'future','nuclear power')

#Connect to twitter endpoint
search_url = "https://api.twitter.com/2/tweets/search/recent"
json_response = dm.connect_to_endpoint(search_url, qp)
        
# Output files
json_file ='tweets.json'
csv_file = 'Tweets.csv' 
output_directory = r'./data'
        
# Building a JSON file usin JSON response.
with open(json_file, 'w') as file :
    json.dump(json_response, file)

# Converting json to csv file.
csv_path = os.path.join(output_directory, csv_file)
dm.json_to_csv('tweets.json',csv_path)
print("csv file is saved successfully as \"Tweets.csv\". ")
print(f"Saved directory : {csv_path}")