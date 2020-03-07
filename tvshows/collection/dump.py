import settings
import tweepy
import dataset
from textblob import TextBlob
from datafreeze import freeze

db = dataset.connect(settings.CONNECTION_STRING)

result = db[settings.TABLE_NAME].all()
freeze(result, format='csv', filename=settings.CSV_NAME)
