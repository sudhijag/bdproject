from collections import Counter
import numpy as np
from scipy.special import softmax
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

url_path = "https://raw.githubusercontent.com/mandalbiswadip/data_storage/main/log_regression_weights/"

stop_words=set(stopwords.words('english'))


unique_words = None
W = None
idf_dict = None
word_index_dict = None

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
# 	print("TWEET ", tweet)
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
# 			print(el)
			cleantokens2.append(el[1:])

# 	print("TWEET after removing", cleantokens2)

	punct_list = list(string.punctuation)
# 	print("PUNCT", string.punctuation)
	for el in cleantokens2:
		#remove punc
		for punc in punct_list:
			if punc in el:
				el = el.replace(punc, ' ')
			el= el.strip()

# 	print("TWEET after removing", cleantokens2)

	#two options: exists, counts in map

	#remove stopwords
	cleantokens3= []
	for el in cleantokens2:
		if el not in STOP_WORDS and el not in stop_words:
			cleantokens3.append(el)

# 	print("TWEET after removing", cleantokens3)
	#first 1 gram then 2 gram

	
	return " ".join(cleantokens3)


def read_file(filename):
	import requests

	response = requests.get(f"{url_path}/{filename}")
	data = response.text
	return data

def load_idf_dict():
    return {line.split()[0]: float(line.split()[1]) for line in read_file("idf_dict.txt").splitlines()}

def load_word_list():
    return read_file("vocab.txt").splitlines()

def load_weights():
    weight_text = read_file("weights.txt")
    return np.array(
        [[float(y) for y in x.split("\t")] for x in weight_text.splitlines()]
    )


def load_model():
	global unique_words
	global W
	global idf_dict
	global word_index_dict
	unique_words = load_word_list()
	W = load_weights()
	idf_dict = load_idf_dict()


	word_index_dict = dict()
	for index, word in enumerate(unique_words):
		word_index_dict[word] = index

def get_tfidf(sentence):
    words = nltk.word_tokenize(sentence)
    vec = np.zeros(len(unique_words) + 1)
    vec[-1] = 1 # this is for the bias term

    for word, v in Counter(words).items():
        if word in word_index_dict:
            vec[word_index_dict[word]] = v * idf_dict[word]
    return vec

predict = lambda x: np.argmax(softmax(- x @ W))

def get_prediction(tweet):
	global unique_words
	global W
	global idf_dict
	global word_index_dict

	if unique_words is None:
		load_model()
	print(tweet)
	feature = get_tfidf(preprocessing_tweets(tweet))
	return predict(feature)

if __name__ == "__main__":

	tweet = "Twitter is bought by elon musk"
	print(get_prediction(tweet))