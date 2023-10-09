# NLP-package
NLP-package is a comprehensive toolkit for classification problem. 
This package provides various types of tools, including classical machine learning models, neural networks, and large language models.
In our particular case, we used those tool for sentiment detection, but users can leverage this for any classification problems.

# Data mining 
Data mining module provides function for collecting tweets from X('Twitter'). 
Since early this year, data scraping is monetized. In order to use this module, you need to purchase API beforehand. 

# Data Labeling
Data labeling is the largest bottleneck of supervised learning. To deal with this problem, our team selected 7 lexicon-based-sentiment analysis libraries so that we can get reasonably reliable labels for the tweets. Though manual labeling would be the most precise way to do it, but this module significantly saves your time and resource. Our labeling module will provide two different labeling methods. 
1. Averaging method
    Among 7 tools, 4 libraries provide sentiment polarity scores. Others don't. Polarity score ranges from -1 to 1, -1 being negative and 1 being positive. The libraries that do not offer score will return either -1 or 1. One intuitive way to label the tweets is averaging all the numbers obtained from libraries. 

2. Majority voting method
    The other way we can think of is voting. We have odd number(7) of tools. Each tool will cast a vote on the sentiment of text. However, even with majority vote, some tweets will not be labeled. Suppose 3 labeled as 'Negative', 3 labeled as 'Positive' and 1 labeled as 'Neutral'. In this case, this module takes average score into account. Final label will be determined by average score. 