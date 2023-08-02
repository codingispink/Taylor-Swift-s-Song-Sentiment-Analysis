# Taylor Swift's Songs Sentiment Analysis 

As a Swfitie, this has been so far my favorite project to work on. In this project, we will analyze the sentiment as well as lyrics of Taylor Swift's songs throughout the past 15 years of her careers. Thank you Jan Llenzl Dagohoy for the datasets and Codeacademy for the instructions/inspiration. Let's get into it.
### Installation
These are the packages to install beforehand:
```
import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
```
**Dataset**
Attached to the file tree are two datasets called: *Taylor Swift Lyrics 2006-2002.csv* and *album_year_name.csv* (Shout out to JAN LLENZL DAGOHOY) [1] Download these files to prepare for the project.

### HANDS-ON

**DATA CLEANING**

**1. Add years to all of the album name:**
   ```
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

    lyrics['album_year'] = lyrics.apply(lambda row: album_release(row), axis=1)
    lyrics.head()
   ```
**2. Make all words lowercase** 

```
lyrics['clean_lyric'] = lyrics['lyric'].str.lower()
```

**3. Remove all the "s" and stop words**

'''
lyrics['clean_lyric']= lyrics['clean_lyric'].str.replace('[^\w\s]','')
stop = ['the', 'a', 'this', 'that', 'to', 'is', 'am', 'was', 'were', 'be', 'are', 'is']
lyrics['clean_lyric'] = lyrics['clean_lyric'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
from sklearn.feature_extraction import text
skl_stop = text.ENGLISH_STOP_WORDS
'''

### Find Keyword Mentions

In this project, let's look at 3 words: 'midnight', 'time', 'place'. Feel free to change it to words you want to. What you can do here is that you will access all the words in the clean_lyric column and only get those that are 'midnights'. Do similarly with 'time' and 'place'. You can also use the 'sum' function to count the amount of time a specific word is mentioned.

```
lyrics['midnight']=lyrics['clean_lyric'].str.contains('midnight')
print(sum(lyrics['midnight']))
```

This case describes a scenario when you want to analyze a specific word. What if you want to analyze a list of words that are very similar in sentiments? For example, love? In this case, I will create a list of all the words that describes the feeling of being in love (thank you merriam-webster dictionary). I am also interested in knowing how often her songs are about her contemplations of growth and experiences. Or just as a sad song queen myself, how often her songs depict feelings of a past relationship.

```
breakup =['tears', 'break', 'shatter', 'heartbreak', 'pain', 'hurt', 'sad', 'dark', 'tears', 'cut', 'miss', 'cry', 'cheat', 'betray']
inlove = ['adore', 'like' ,'fancy', 'affection', 'worship', 'baby', 'babe', 'honey', 'passion', 'dream', 'marry', 'darling', 'boyfriend', 'love']
think = ['think', 'thought', 'reflect', 'why', 'remember', 'miss', 'look', 'mean', 'evil', 'monster', 'bone', 'say']
```
Next, I will join the lists into a regular expression string using the .join() function and the "|" to indicate "or". Finally, I can count the amount of time these words are mentioned throughout her song.

```
breakup_regex = '|'.join(breakup)
inlove_regex = '|'.join(inlove)
think_regex = '|'.join(think)

lyrics['breakup'] = lyrics['clean_lyric'].str.contains(breakup_regex)
lyrics['inlove'] = lyrics['clean_lyric'].str.contains(inlove_regex)
lyrics['think'] = lyrics['clean_lyric'].str.contains(think_regex)

print("breakup words: ", breakup_count)
print("in love words: ",inlove_count)
print("think words: ", think_count)
```

The results should be like this. It does seem like Taylor spent lots of time thinking and contemplating in her songs. Her songs, at least in this analysis, are usually about happy lovey dovey relationships as well. 

breakup words:  374
in love words:  1240
think words:  1064

### Analyze Taylor Swift's Album Over the Year
Based on the Analysis above, let's see which one is the love album based on how many words associated with love are counted in that album. We can do that through sorting. Make sure you read the album_year_name first (since the first dataset does not include release year of each album). In this part, you will sort these values by the 'inlove' list in the descending order.

```
year_name = pd.read_csv('album_year_name.csv')
ym['album_name'] = year_name['album_name']
ym.sort_values(by='album_year', ascending=True, inplace=True)
year_name.sort_values(by='album_year', ascending=True, inplace=True)
print("\tTaylor Swift's most in-love album ranking from most to least: ") #this will give you a nice title for the list
print(ym.sort_values(by='inlove', ascending=False))
```

The result shows that the most in-love album is : Lover. This makes sense to me personally as a Swiftie. Lover is an album that, in her onw words, describes "all types of love" and depicts her happiest time in her relationship. 

Following similar steps, you can find out which one is the breakup album too. The result shows that the breakup album is: Red. If you are a Swiftie, you know that Red is so breakup coded. [3] A few iconic songs in this album are: All Too Well (10-minute version) *screaming and crying*, I Almost Do, We Are Never Getting Back Together...

### Visualize the changes in lyrics of Taylor Swift's songs.
We can compare the amount of time 'love' and 'breakup' words are mentioned through her song over time.
```
plt.plot(ym['album_year'], ym['inlove'], label = 'in-love')
plt.plot(ym['album_year'], ym['breakup'], label = 'breakup')
plt.title('Taylor Swift mentioned of love and breakup')
plt.legend()
plt.show()
```
![Alt text](/loveandbreakupthroughtime.png?raw=true "loveandbreakupthroughtime")

### Lyrics Sentiment Analysis
First, download a package from NLTK

```
nltk.download('vader_lexicon')
```

Let's first try to see how sentiment analysis work using SentimentIntensityAnalyzer package.

```
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("I love Taylor Swift")
```
This line should produce something like this: {'neg': 0.0, 'neu': 0.143, 'pos': 0.857, 'compound': 0.7184}. This interprets that the line "I love Taylor Swift" is much more positive than negative. Please note that 'compound' is a score ranging between [-1, 1] tells you whether a line/word is more positive or negative. The higher the score is, the more positive it is.

Create a new column called polarity and apply the Sentiment Intensity Analyzer method to the clean_lyric column with a lambda expression. Then, transform the polarity dictionary into columns of the DataFrame. This will first create a new column called polarity with the polarity dictionary in it. It calculates all the neg, pos, compound of each lyric in the dataframe. Then, applying pd.Series will enable us to have each of the component of polarity as its own column. We will eventually drop the polarity column.
```
lyrics['polarity'] = lyrics['clean_lyric'].apply(lambda x: sia.polarity_scores(x))
lyrics[['neg', 'neu', 'pos', 'compound']] = lyrics['polarity'].apply(pd.Series)
lyrics.drop('polarity', axis=1)
```


**Let's Visualize the Sentiment Analysis of Taylor Swift's songs over the year **
```
yearly_sentiment = lyrics.groupby('album_year').sum().reset_index()
plt.plot(yearly_sentiment['album_year'], yearly_sentiment['compound'])
plt.title("Taylor Swift's average Album Sentiment")
plt.show()
```
![Alt text](/averagesentiment.png?raw=true "averagesentiment.png")

### Compare in-love and break-up mentions in Taylor Swift's songs. Which one are more positive or negative?
Let's compare the difference between mentions of 'day' and 'night' in her music to see which one is more positive and negative. What we do is that we will create a dataframe filtered out only inlove and breakup. We can then calculate the sentiment of each dataframe from the compound values.

```
inlove = lyrics[lyrics['inlove']==True]
breakup = lyrics[lyrics['breakup']==True]
print("In love: ", len(inlove))
print("Break Up: ", len(breakup))
inlove_sentiment = inlove['compound'].sum()
breakup_sentiment = breakup['compound'].sum()
print("inlove sentiment: ", inlove_sentiment)
print("breakup sentiment: ", breakup_sentiment)
```
The result shows that inlove sentiment is 356.7 and breakup sentiment is -40.91, which means that inlove is more positive when breakup is really negative. This makes sense.

## ACKNOWLEDGEMENT
[1] Jan LLenzzl Dagohoy: https://www.kaggle.com/datasets/thespacefreak/taylor-swift-song-lyrics-all-albums

[2] Codeacademy: https://www.youtube.com/watch?v=rcmOa9c874s&t=1534s&ab_channel=Codecademy

[3] American Songwriter: https://americansongwriter.com/5-stellar-breakup-albums/#:~:text=Red%20%E2%80%93%20Taylor%20Swift,department%20has%20long%20been%20cemented. 

[4] Merriam-Webster: https://www.merriam-webster.com/thesaurus/love
