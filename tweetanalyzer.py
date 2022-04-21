#homework3-2

import random
import tweepy
import time
import kafka
from kafka import KafkaProducer, KafkaConsumer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import udf

from pyspark.sql.types import LongType
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from textblob import TextBlob
import sys

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import Stream

import string
import numpy as np

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


def main():
	if len(sys.argv) != 5:
	    print("Usage: tweetanalyzer.py <bootstrap-servers> <checkpoint-dir> <input-topic> <output-topic>", file=sys.stderr)
	    sys.exit(-1)

	bootstrapServers = sys.argv[1]
	checkpointDir = sys.argv[2]
	inputtopic= sys.argv[3]
	outputtopic= sys.argv[4]
	subscribeType="subscribe"

	spark = SparkSession.builder.appName("Homework3").getOrCreate()
	spark.sparkContext.setLogLevel("ERROR")

	#read in the file and begin
	coeff_file = open("coeff.txt")
	coeff_dict = {}
	for line in coeff_file:
	    key, value = line.split()
	    coeff_dict[key] = value

	print(coeff_dict)

	lines = spark.readStream.format("kafka").option("kafka.bootstrap.servers", bootstrapServers)\
	    .option(subscribeType, inputtopic)\
	    .option("failOnDataLoss", "false")\
	    .load()\
	    .selectExpr("CAST(value AS STRING)")

	def LogisticRegression(tweet_text):
		#create an rdd from tweet text
		mylist= tweet_text.split(' ')
		ans=0
		for word in mylist:
			if(word in coeff_dict):
				ans += float(coeff_dict[word])

		return str(ans)

	def L1LogisticRegression(tweet_text):
		mylist= tweet_text.split(' ')
		ans=0
		for word in mylist:
			if(word in coeff_dict):
				ans += float(coeff_dict[word])

		return str(ans)

	def L2LogisticRegression(tweet_text):
		mylist= tweet_text.split(' ')
		ans=0
		for word in mylist:
			if(word in coeff_dict):
				ans += float(coeff_dict[word])

		return str(ans)

	def preprocessing_tweets(tweet) :
		print("TWEET ", tweet)
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
				print(el)
				cleantokens2.append(el[1:])

		#print("TWEET after removing", cleantokens2)

		punct_list = list(string.punctuation)
		print("PUNCT", string.punctuation)
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
		return ' '.join(cleantokens3)

	def NaiveBayes(tweet_text):
		return 0

	lr_udf= udf(LogisticRegression, StringType())
	l1lr_udf= udf(L1LogisticRegression, StringType())
	l2lr_udf= udf(L2LogisticRegression, StringType())
	nb_udf= udf(NaiveBayes, StringType())
	clean_udf= udf(preprocessing_tweets, StringType())

	lines = lines.withColumn("CLEANED", clean_udf(lines['value']))
	lines = lines.withColumn("LR", lr_udf(lines['CLEANED']))
	lines = lines.withColumn("L1LR", l1lr_udf(lines['CLEANED']))
	lines = lines.withColumn("L2LR", l2lr_udf(lines['CLEANED']))
	lines = lines.withColumn("NB", nb_udf(lines['CLEANED']))
	#lines = lines.withColumnRenamed("value","msg").withColumnRenamed("sentiment","value")


	query = lines.writeStream.outputMode('append').format('console').start().awaitTermination()
	#query = lines.writeStream.format("kafka").outputMode('append').\
	#option("checkpointLocation", checkpointDir).\
	#option("kafka.bootstrap.servers", bootstrapServers).option("topic", outputtopic).start().awaitTermination()

if __name__ == "__main__":
    main()