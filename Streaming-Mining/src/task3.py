import tweepy
import random
import sys


TEXT = "text"
HASHTAGS = "hashtags"
SQNO = -1
reservoir_size = 100
tags = {}
tweets = []

# Local path
# output_file = "data/HW6/task3_output.csv"

port = int(sys.argv[1])
output_file = sys.argv[2]

f= open(output_file, "w")

class MyStreamListener(tweepy.StreamListener):
    def on_status(self, tweet):
        global SQNO
        global tweets
        global tags

        hashtags = tweet.entities['hashtags']

        if (len(hashtags) > 0):

            SQNO = SQNO + 1
            if (SQNO < reservoir_size):
                tweets.append(tweet)
                for tag in hashtags:
                    hashtag_string = tag['text']
                    if (hashtag_string in tags.keys()):
                        tags[hashtag_string] = tags[hashtag_string] + 1
                    else:
                        tags[hashtag_string] = 1

            else:
                r = random.randint(1, SQNO)
                if (r <= 100):
                    remove_pos = random.randint(0, reservoir_size - 1)
                    remove_tweet = tweets[remove_pos]

                    # remove old hashtags
                    remove_tags = remove_tweet.entities[HASHTAGS]
                    for tag in remove_tags:
                        T1 = tag[TEXT]
                        tags[T1] = tags[T1] - 1
                        if (tags[T1] == 0):
                            del tags[T1]

                    # add new tweet to old position
                    tweets[remove_pos] = tweet

                    # add new hashtags
                    add_tags = tweet.entities[HASHTAGS]
                    for tag in add_tags:
                        T2 = tag[TEXT]
                        if (T2 in tags.keys()):
                            tags[T2] = tags[T2] + 1
                        else:
                            tags[T2] = 1
                else:
                    pass

            results_tags = []

            for tag, frequency in tags.items():
                entry = [tag, frequency]
                results_tags.append(entry)
            results_tags.sort(key=lambda x: x[0])
            results_tags.sort(key=lambda x: x[1], reverse=True)

            ctr = 0
            prev = results_tags[0][1]

            f = open(output_file, "a")
            f.write("The number of tweets with tags from the beginning: " + str(SQNO + 1) + "\n")
            for tag in results_tags:

                if (tag[1] == prev):
                    f.write(str(tag[0])+": "+str(tag[1])+ "\n")
                    print(tag)
                else:
                    if (ctr < 2):
                        prev = tag[1]
                        ctr = ctr + 1
                        print(tag)
                        f.write(str(tag[0]) + ": " + str(tag[1])+"\n")
                    else:
                        break

            f.write("\n")
            f.close()

    def on_error(self, status_code):
        if status_code == 420:
            return False


consumer_token = "PoK4wUVmAF0CEjoUPO2Zpl3Vz"
consumer_secret = "BmUZL2PlZntOMJvwCuAWX0Idx3Rbbvi6nDhgAERZ9Ki0tnwv11"

access_token = "1105117550587764738-zohmWEhfLXQSYQrFjzo6DCO07R76Dy"
access_token_secret = "QVYNKL0onlGmEjgpJB768V9H2kaBAmja68Ew9EBixR9pH"

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

stream_listener = MyStreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)

stream.filter(track=['covid'])