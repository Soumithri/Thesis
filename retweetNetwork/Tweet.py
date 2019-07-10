import datetime
from enum import Enum


class TweetType(Enum):
    RETWEET = 1
    QUOTE = 2
    REPLY = 3
    MENTION = 4


class Tweet:
    MENTION = "user_mentions"
    QUOTE_FIELD = "quoted_status"
    REPLY_FIELD = "in_reply_to_user_id"
    RETWEET_FIELD = "retweeted_status"

    def __init__(self, tweet):
        self.tweet = tweet
        self.author_id = int(self.tweet['user']['id'])
        self.create_time = int(
            datetime.datetime.strptime(self.tweet['created_at'], "%a %b %d %H:%M:%S %z %Y").timestamp())

        # Retweet field
        if Tweet.RETWEET_FIELD in tweet:
            self.retweet = True
            self.retweet_author_id = int(tweet[Tweet.RETWEET_FIELD]["user"]["id"])
            self.retweet_mentions = [int(user['id']) for user in tweet[Tweet.RETWEET_FIELD]["entities"][Tweet.MENTION]]
            self.retweet_create_time = int(
            datetime.datetime.strptime(self.tweet[Tweet.RETWEET_FIELD]['created_at'], 
                                       "%a %b %d %H:%M:%S %z %Y").timestamp())
        else:
            self.retweet = False
            self.retweet_author_id = None
            self.retweet_mentions = None
            self.retweet_create_time = None

        # Quote field
        if Tweet.QUOTE_FIELD in tweet:
            self.quote = True
            self.quote_author_id = int(tweet[Tweet.QUOTE_FIELD]["user"]["id"])
            self.quote_mentions = [int(user['id']) for user in tweet[Tweet.QUOTE_FIELD]["entities"][Tweet.MENTION]]
        else:
            self.quote = False
            self.quote_author_id = None

        # Mention field
        if Tweet.MENTION in tweet["entities"]:
            self.mentions = [int(user['id']) for user in tweet["entities"][Tweet.MENTION]]
        else:
            self.mentions = None

        # Reply field
        if Tweet.REPLY_FIELD in tweet and tweet[Tweet.REPLY_FIELD]:
            self.reply_id = int(tweet[Tweet.REPLY_FIELD])
        else:
            self.reply_id = None

