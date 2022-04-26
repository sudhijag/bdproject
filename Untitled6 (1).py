#!/usr/bin/env python
# coding: utf-8

# In[30]:


import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOP_WORDS = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at
back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by
call can cannot ca could
did do does doing done down due during
each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except
few fifteen fifty first five for former formerly forty four from front full
further
get give go
had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred
i if in indeed into is it its itself
keep
last latter latterly least less
just
made make many may me meanwhile might mine more moreover most mostly move much
must my myself
name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere
of off often on once one only onto or other others otherwise our ours ourselves
out over own
part per perhaps please put
quite
rather re really regarding
same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such
take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two
under until up unless upon us used using
various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would
yet you your yours yourself yourselves
""".split()
)


def preprocessing_tweets(tweet) :
	#print("TWEET ", tweet)
	tweet= tweet.lower()
	tweet=tweet.strip()

	arr= tweet.split(" ")

	#remove url
	cleantokens= []
	for el in arr:
		if(not 'http' in el):
			el= el.strip()
			cleantokens.append(el)


	#remove usernames, keep after hashtag
	cleantokens2= []
	for el in cleantokens:
		if(el.find("@") != 0 and not el.find("#") == 0): #regular words, not hashtags
			cleantokens2.append(el)
		if(el.find("#") == 0):
			#print(el)
			cleantokens2.append(el[1:])

	#print("TWEET after removing", cleantokens2)

	punct_list = list(string.punctuation)
	#print("PUNCT", string.punctuation)
	for el in cleantokens2:
		#remove punc
		for punc in punct_list:
			if punc in el:
				el = el.replace(punc, ' ')
			el= el.strip()

	#print("TWEET after removing", cleantokens2)

	#two options: exists, counts in map

	#remove stopwords
	cleantokens3= []
	for el in cleantokens2:
		if el not in STOP_WORDS:
			cleantokens3.append(el)

	#print("TWEET after removing", cleantokens3)
	#first 1 gram then 2 gram

	
	return cleantokens3


tweet= " @JReebo: Who wants to get there nose in these bad bois then #scally #chav #sockfetish #stinking http://t.co/FeQxgN0W6I hot sox and legs"
preprocessing_tweets(tweet)


# In[ ]:


tweet= " @JReebo: Who wants to get there nose in these bad bois then #scally #chav #sockfetish #stinking http://t.co/FeQxgN0W6I hot sox and legs"
preprocessing_tweets(tweet)


# In[32]:


import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings


# In[35]:


def remove_hyperlinks_marks_styles(tweet):
    
    # remove old style retweet text "RT"
    new_tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    new_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', new_tweet)

    # remove hashtags
    # only removing the hash # sign from the word
    new_tweet = re.sub(r'#', '', new_tweet)
    
    return new_tweet


# In[34]:


# instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

def tokenize_tweet(tweet):
    
    tweet_tokens = tokenizer.tokenize(tweet)
    
    return tweet_tokens


# In[36]:


stemmer = PorterStemmer()

def get_stem(tweets_clean):
    
    tweets_stem = []
    
    for word in tweets_clean:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)
        
    return tweets_stem


# In[39]:


def process_tweet(tweet):
    
    processed_tweet = remove_hyperlinks_marks_styles(tweet)
    tweet_tokens = tokenize_tweet(processed_tweet)
    tweets_clean = remove_stopwords_punctuations(tweet_tokens)
    tweets_stem = get_stem(tweets_clean)
    
    return tweets_stem


# In[33]:


nltk.download('stopwords')

#Import the english stop words list from NLTK
stopwords_english = stopwords.words('english')

punctuations = string.punctuation

def remove_stopwords_punctuations(tweet_tokens):
    
    tweets_clean = []
    
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in punctuations):
            tweets_clean.append(word)
            
    return tweets_clean


# In[329]:


import csv

columns = defaultdict(list) # each value in each column is appended to a list

with open('labeled_data.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

#print(columns['tweet'])
#print(columns['class'])

tweets = columns['tweet']
sentiments = columns['class'] 


#print(tweets[:5])
#print(sentiments[:5])


freq_d = term_frequency(tweets[:2000], sentiments[:2000])
#print(get_frequency(freq_d,'rt',2))
#print(freq_d)


#print('yeeeeeeeeeee '+ str(freq_d.get(('rt', '2'),3)))

nb_training(tweets[:2000],sentiments[:2000])


# In[65]:


import numpy as np
import re
import string


# In[270]:


def term_frequency(tweets, sentiment):
    
    frequency = {}
    
    
    for x, y in zip(tweets, sentiment):
        for word in process_tweet(x):
            
            sentimentCount = (word, y)
            
            if sentimentCount in frequency:
                frequency[sentimentCount] += 1
            else:
                frequency[sentimentCount] = frequency.get(sentimentCount, 1)
                
    return frequency


# In[315]:


tweets = ['hello the world', 'chrome file edit', 'note to self', 'open open open', ' next note self']
sentiments = [0,1,0,1,1]

term = term_frequency(tweets, sentiments)
print(term)
d = 'open'
x = term.get((d,1), 5)
print(type(term))
print(x)


print(get_frequency(term,'world',0))


# In[302]:


def get_frequency(x, y, z):
    freqency = x.get((y,z), 0)
    return freqency


# In[328]:


import json
def nb_training(tweetList, sentimentList):
    
    frequency = {}
    
    
    for x, y in zip(tweetList, sentimentList):
        for word in process_tweet(x):
            
            sentimentCount = (word, y)
            
            if sentimentCount in frequency:
                frequency[sentimentCount] += 1
            else:
                frequency[sentimentCount] = frequency.get(sentimentCount, 1)
    
    
    
    offensiveTerms = 1
    hateTerms = 1
    regTerms = 1
    #get total nmber of times a word is in a regular tweet vs a hateful tweet
    
    for sentiments in frequency.keys():
        
        #print(sentiments[1])
        if int(sentiments[1]) == 2:
            regTerms += frequency[sentiments]
        elif int(sentiments[1]) == 1:
            offensiveTerms += frequency[sentiments]
        elif int(sentiments[1]) == 0:
            hateTerms += frequency[sentiments]
    
    print(regTerms)
    print(offensiveTerms)
    #Tweet totals
    totalTweets = 0
    regTweets = 0
    hateTweets = 0
    offensiveTweets = 0
    
    for sentiment in sentimentList:
        totalTweets +=1
        #print(type(sentiment))
        if int(sentiment) == 2:
            regTweets +=1
        elif int(sentiment) == 1:
            offensiveTweets +=1
        elif int(sentiment) == 0:
            hateTweets +=1  
            
    print(totalTweets)
    print(regTweets)
    print(hateTweets)
    print(offensiveTweets)
    hatePrior = hateTweets / totalTweets
    regularPrior = regTweets / totalTweets
    offensivePrior = offensiveTweets / totalTweets
    
    regWordProbability = {}
    offensiveProbability = {}
    hateWordProbablity = {}
    
    x = term.get(('rt',2), 0)
    print(x)
    #https://stackoverflow.com/questions/22978602/how-to-return-unique-words-from-the-text-file-using-python
    wordSet = set([sentiments[0] for sentiments in frequency.keys()])
    #print(wordSet)
    # have an alpha value in case value of frequency is zero 
    alpha = 1
    for x in wordSet:

       
        regFrequency = frequency.get((x,'2'), 0)
        #print(regFrequency)
        if regFrequency == 0 :
            regWordProbability[x] = alpha / regTerms
        else:
            print('Enter the reg')
            regWordProbability[x] = regFrequency / regTerms
        
        offensiveFrequency = frequency.get((x, '1'), 0)
        if offensiveFrequency == 0 :
            offensiveProbability[x] = alpha / offensiveTerms
        else:
            print('Enter the off')
            offensiveProbability[x] = offensiveFrequency / offensiveTerms
        
        hateFrequency = frequency.get((x, '0'), 0)
        if hateFrequency == 0 :    
            hateWordProbablity[x] = alpha / hateTerms
        else:
            print('Enter the hate')
            hateWordProbablity[x] = hateFrequency / hateTerms
    
    regpItems = regWordProbability.items()
    print(list(regpItems)[:7])
    offpItems = offensiveProbability.items()
    print(list(offpItems)[:7])
    #print(hateWordProbablity[:10])
    
    priors = [str(regularPrior), str(offensivePrior), str(hatePrior)]
    
    
    with open('Prior.txt', 'w') as f:
        f.write('\n'.join(priors))
        f.close()
    
    with open('regWordProbability.txt', 'w') as f:
        json.dump(regWordProbability, f)
        f.close()
        
    with open('offensiveProbability.txt', 'w') as f:
        json.dump(offensiveProbability, f)
        f.close()
    
    with open('hateWordProbablity.txt', 'w') as f:
        json.dump(hateWordProbablity, f)
        f.close()
        
    #return regularPrior, offensivePrior, hatePrior, regWordProbability , offensiveProbability, hateWordProbablity


# In[90]:


def readPriors():
    with open('Prior.txt') as f:
        reg = float(f.readline())
        off = float(f.readline())
        hate = float(f.readline())
        f.close()
    return reg, off, hate
    


# In[116]:


def readFromFile(filename):
    
    txtfile = filename + '.txt'
    with open(txtfile) as f:
        data = f.read()
        f.close()
    
    js = json.loads(data)
  
    return js
    


# In[334]:


nb_predict('@CanIFuckOrNah: What would yall lil ugly bald headed bitches do if they stop making make-up &amp')


# In[333]:


def nb_predict(tweet):
    
    regularPrior, offensivePrior, hatePrior = readPriors()
    regWordProbability = readFromFile('regWordProbability')
    offensiveWordProbablity = readFromFile('offensiveProbability')
    hateWordProbability = readFromFile('hateWordProbablity')
    
    #print(regWordProbability)
    
    tweetList = process_tweet(tweet)
    print(tweetList)
    hateProb = 1
    regProb = 1
    offensiveProb = 1
    
    for x in tweetList:
        #print('hi')
        word = str(x)
        if x in regWordProbability:
            #print(regWordProbability[word])
            regProb = regProb * regWordProbability[word]
        if x in hateWordProbability:
            #print(hateWordProbability[word])
            hateProb = hateProb * hateWordProbability[word]
        if x in offensiveWordProbablity:
            #print(offensiveWordProbablity[word])
            offensiveProb = offensiveProb * offensiveWordProbablity[word]
            
    regProb = regProb * regularPrior
    hateProb = hateProb * hatePrior
    offensiveProb = offensiveProb * offensivePrior
    print(regProb, hateProb, offensiveProb)     
    if regProb == max(regProb, hateProb, offensiveProb):
        return regProb, 2
    elif offensiveProb == max(regProb, hateProb, offensiveProb) :
        return offensiveProb, 1
    else:
        return hateProb, 0


# In[18]:


def writePriors():
    
    regularPrior = 
    offensivePrior = 
    hatePrior = 

    
    
    priors = [str(regularPrior), str(offensivePrior), str(hatePrior)]
    
    with open('test.txt', 'w') as f:
        f.write('\n'.join(priors))
        f.close()
    


# In[ ]:




