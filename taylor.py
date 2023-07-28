#matplotlib in line
import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

lyrics = pd.read_csv("taylor_swift_lyrics_2006-2022_all.csv")
#inspect some first few rows
lyrics.head()
#get info about the dataframe
lyrics.info()

#notice there is no year, so add year
print(lyrics.album_name.unique()) #print album name
#add year
def album_release(row):
    if row['album_name'] == 'Taylor Swift':
        return '2006'
    elif row['album_name'] == 'Speak Now (Deluxe)':
        return '2010'
    elif row['album_name'] == 'Red (Deluxe Edition)':
        return '2012'
    elif row['album_name'] == '1989 (Deluxe)':
        return '2014'
    elif row['album_name'] == 'reputation':
        return '2017'
    elif row['album_name'] == 'Lover':
        return '2019'
    elif row['album_name'] == 'folklore (deluxe version)':
        return '2020'
    elif row['album_name'] == 'evermore (deluxe version)':
        return '2021'
    elif row['album_name'] == "Fearless (Taylor's Version)":
        return '2007'
    elif 'midnights' in row['album_name']:
        return '2022'
    return 'No Date'

 #apply function to the album
lyrics['album_year'] = lyrics.apply(lambda row: album_release(row), axis=1)
lyrics.head()

#clean the lyric text - first thing you do with all text data
#1-lowercase 
lyrics['clean_lyric'] = lyrics['lyric'].str.lower()
lyrics['clean_lyric']= lyrics['clean_lyric'].str.replace('[^\w\s]','')
stop = ['the', 'a', 'this', 'that', 'to', 'is', 'am', 'was', 'were', 'be', 'are', 'is']
lyrics['clean_lyric'] = lyrics['clean_lyric'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#remove illustration column
#lyrics.drop(['clean_lyric_list', 'clean_lyric_list_rejoined'], axis=1, inplace=True)
from sklearn.feature_extraction import text

skl_stop = text.ENGLISH_STOP_WORDS
#FIND KEYWORD MENTIONS
lyrics['midnight']=lyrics['clean_lyric'].str.contains('midnight')
print(sum(lyrics['midnight']))

#love
breakup =['tears', 'break', 'shatter', 'heartbreak', 'pain', 'hurt', 'sad', 'dark', 'tears', 'cut', 'miss', 'cry', 'cheat', 'betray']
inlove = ['adore', 'like' ,'fancy', 'affection', 'worship', 'baby', 'babe', 'honey', 'passion', 'dream', 'marry', 'darling', 'boyfriend', 'love']
think = ['think', 'thought', 'reflect', 'why', 'remember', 'miss', 'look', 'mean', 'evil', 'monster', 'bone', 'say']

breakup_regex = '|'.join(breakup)
inlove_regex = '|'.join(inlove)
think_regex = '|'.join(think)

lyrics['breakup'] = lyrics['clean_lyric'].str.contains(breakup_regex)
lyrics['inlove'] = lyrics['clean_lyric'].str.contains(inlove_regex)
lyrics['think'] = lyrics['clean_lyric'].str.contains(think_regex)

breakup_count = sum(lyrics['breakup'])
inlove_count = sum(lyrics['inlove'])
think_count = sum(lyrics['think'])

print("breakup words: ", breakup_count)
print("in love words: ",inlove_count)
print("think words: ", think_count)

#VISUALIZE HOW TAYLOR SWIFT MENTIONS OF LOVE CHANGED OVER TIME
ym = lyrics.groupby('album_year').sum().reset_index()
print(lyrics['album_year'])
#plot mentions of love over year

plt.plot(ym['album_year'], ym['think'])
plt.title("Taylor Swift Night Mentions")
plt.show()

#WHAT ALBUM IS THE MOST IN LOVE?
year_name = pd.read_csv('album_year_name.csv')
ym['album_name'] = year_name['album_name']
ym.sort_values(by='album_year', ascending=True, inplace=True)
year_name.sort_values(by='album_year', ascending=True, inplace=True)
print("\tTaylor Swift's most in-love album ranking from most to least: ")
print(ym.sort_values(by='inlove', ascending=False))
#WHAT ALBUM IS THE BREAKUP ALBUM?
bu = ym.sort_values(by=['breakup'], ascending = False)
print("\tTaylor Swift's most breakup album ranking from most to least: ")
print(bu.drop(columns=['track_n','midnight']))
#WHAT DOES SHE THINK THE MOST ABOUT?
th = ym.sort_values(by='think', ascending = False)
print("\tTaylor Swift's most contemplative album ranking from most to least: ")
print(th.drop(columns=['track_n', 'midnight', 'line']))

#COMPARE IN-LOVE WITH BREAKUPS
plt.plot(ym['album_year'], ym['inlove'], label = 'in-love')
plt.plot(ym['album_year'], ym['breakup'], label = 'breakup')
plt.title('Taylor Swift mentioned of love and breakup')
plt.legend()
plt.show()
                            
#TASK 3: TOKENIZE THE WORD:
#run this cell to tokenize the words in the clean_lyric column
lyrics['lyrics_tok'] = lyrics['clean_lyric'].str.split(' ')
print(lyrics.head())

#determine most frequently used words:
#create a list of all the word in the lyric_tok
word_list = [word for list_ in lyrics['lyrics_tok'] for word in list_]

#count number of times each word appears
word_frequency = collections.Counter(word_list)
#sort word frquencies to most used to least used
word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
print(word_frequency)

#LYRICS SENTIMENT ANALYSIS
#add a package from NLTK
nltk.download('vader_lexicon')
#run to see how sentiment analysis works
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("I love Taylor Swift")
#create a new column called polarity and apply the sia method to the clean lyrics
lyrics['polarity'] = lyrics['clean_lyric'].apply(lambda x: sia.polarity_scores(x))
#run this cell to transform the popularity dictionary into columns of
lyrics[['neg', 'neu', 'pos', 'compound']] = lyrics['polarity'].apply(pd.Series)
lyrics.drop('polarity', axis=1)

#CORPUS SENTIMENT ANALYSIS
#calculate overall sentiment for pros, neg, sentiment
pos = sum(lyrics['pos'])
neg = sum(lyrics['neg'])
compound = sum(lyrics['compound'])
#Print the overall sentiments
print("positive: ", pos)
print('negative: ', neg)
print('compound: ', compound)

#a new dataframe using the groupby method for the album year
yearly_sentiment = lyrics.groupby('album_year').sum().reset_index()
#visualize
plt.plot(yearly_sentiment['album_year'], yearly_sentiment['compound'])
plt.title("Taylor Swift's average Album Sentiment")
plt.show()

#is love  more positive or negative?
#dataframe with night
lyrics['time']=lyrics['clean_lyric'].str.contains('time')
lyrics['place']=lyrics['clean_lyric'].str.contains('place')
time = lyrics[lyrics['time']==True]
place = lyrics[lyrics['place']==True]
print("time: ", len(time))
print("place: ", len(place))
love_sentiment = time['compound'].sum()
pain_sentiment = place['compound'].sum()
print("time sentiment: ", love_sentiment)
print("place sentiment: ", pain_sentiment)
























