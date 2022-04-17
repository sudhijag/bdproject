from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import Stream
import logging
import json

logging.basicConfig(filename="twKafkaApp.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.info('Started')

#consumer key, consumer secret, access token, access secret.
consumerKey= "6wiX7IqnZKHetMISzcYIsTl3l"
consumerSecret= "FhzZ8oGOYtCs0gl1LqbJcldCDwOodaDzlVeG4XTa9AjjhwZ4z2"
accessToken= "2182214210-PmRGeJVKPQnDz7ft7thPHdGVH94sqiur5YEjPKj"
accessSecret= "TrqU8tZlIhTK0bMCZI3WgnVWsbnkLZoH02jVdyRS0jzpk"

class listener(Stream):
    def on_data(self, data):
        try:
            # read the Twitter data which comes as a JSON format
            msg = json.loads(data)

            # the 'text' in the JSON file contains the actual tweet.
            print(msg['text'].encode('utf-8'))

            # the actual tweet data is sent to the client socket
            #self.client_socket.send(msg['text'].encode('utf-8'))
            return True

        except BaseException as e:
            # Error handling
            print("Ahh! Look what is wrong : %s" % str(e))
            return True
 
    def on_error(self, status):
        print("ERROR HERE")
        return True

    """def on_data(self, data):
        print(data.text)
        return(True)

    def on_error(self, status):
        print(status)"""

#auth = OAuthHandler(consumerKey, consumerSecret)
#auth.set_access_token(accessToken, accessSecret)

twitterStream = listener(consumer_key= consumerKey, consumer_secret= consumerSecret, access_token= accessToken,access_token_secret= accessSecret)

#listofcontroversialtopics= []
twitterStream.filter(track=["car"], languages=["en"]) #TODO: add query rules, remove non-english tweets

logging.info('Finished')