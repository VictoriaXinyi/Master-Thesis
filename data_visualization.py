#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:55:09 2019

@author: sunqiaoyubing
"""
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#get_ipython().run_line_magic('matplotlib', 'inline')
#
import warnings
warnings.filterwarnings('ignore')

#################################################################################
## 1. User level data Visulization
user = pd.read_csv("/Users/sunqiaoyubing/Desktop/user.csv", encoding='latin-1')

#1.
X_1 = user.drop(['id'], axis=1)
sns.pairplot(X_1,palette="husl", hue='Bot',markers=["o", "+"]);
             
#2.
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(X_1.corr(), cmap='coolwarm', annot=True, linewidths=.5, fmt= '.1f',ax =ax,square=True)

#3.
df1 = user[['name_length','Bot']]
bot_ = df1.ix[(df1['Bot']==1)]
Nonbot_ = df1.ix[(df1['Bot']==0)]
from scipy.stats import norm
ax1 = sns.distplot(bot_['name_length'], fit=norm, kde=False,label="Bots")
ax2 = sns.distplot(Nonbot_['name_length'], fit=norm, kde=False,label="Humans")
plt.title('Distribution of Name length for bot and humas')
plt.legend()

#4.
df1 = user[['sn_length','Bot']]
bot_ = df1.ix[(df1['Bot']==1)]
Nonbot_ = df1.ix[(df1['Bot']==0)]
from scipy.stats import norm
ax1 = sns.distplot(bot_['sn_length'], fit=norm, kde=False,label="Bots")
ax2 = sns.distplot(Nonbot_['sn_length'], fit=norm, kde=False,label="Humans")
plt.title('Distribution of screen Name length for bot and humas')
plt.legend()

#5.
df1 = user[['desc_length','Bot']]
bot_ = df1.ix[(df1['Bot']==1)]
Nonbot_ = df1.ix[(df1['Bot']==0)]
from scipy.stats import norm
ax1 = sns.distplot(bot_['desc_length'], fit=norm, kde=False,label="Bots")
ax2 = sns.distplot(Nonbot_['desc_length'], fit=norm, kde=False,label="Humans")
plt.title('Distribution of description length for bot and humas')
plt.legend()

#6.
plt.title('Bot count between geo_enabled', size=20, y=1.1)
sns.countplot(x = 'Bot', hue='geo_enabled', data=user)

#7.
plt.title('Bot count between default_profile', size=20, y=1.1)
sns.countplot(x = 'Bot', hue='default_profile', data=user)

#8.
plt.title('Bot count between profile_use_background_image', size=20, y=1.1)
sns.countplot(x = 'Bot', hue='profile_use_background_image', data=user)

#9.
plt.title('Bot count between profile_background_tile', size=20, y=1.1)
sns.countplot(x = 'Bot', hue='profile_background_tile', data=user)

#10.
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))

bins = 2

ax1.scatter(user.listed_count[user.Bot == 1],user.favourites_count[user.Bot == 1],c='r')
ax1.set_title('Frequency on Favorites count --- Bot',size=14)

ax2.scatter(user.listed_count[user.Bot == 0],user.favourites_count[user.Bot == 0],c='g')
ax2.set_title('Frequency on Favorites count --- Human',size=14)

plt.xlabel('Favourites_count')
plt.ylabel('Frequency')
plt.show()

#11.
df1 = user[['listed_count','Bot']]
bot_ = df1.ix[(df1['Bot']==1)]
Nonbot_ = df1.ix[(df1['Bot']==0)]
from scipy.stats import norm
ax1 = sns.distplot(bot_['listed_count'], fit=norm, kde=False,label="Bots")
ax2 = sns.distplot(Nonbot_['listed_count'], fit=norm, kde=False,label="Humans")
plt.title('Distribution of listed count for bot and humas')
plt.legend()

#12.
sns.set(style="ticks", context="talk")
pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)
g = sns.lmplot(x="friends_count", y="favourites_count", hue="Bot", data=user,
               palette=pal, size=7)
g.set_axis_labels("friends_count", "favourites_count")


##################################################################################################
## 2. User_Tweets data visulization
Tweets_pro = pd.read_csv('/Users/sunqiaoyubing/Desktop/Tweets_pro.csv',sep=";")
Tweetsle = pd.read_csv('/Users/sunqiaoyubing/Desktop/Tweetsle.csv',sep=";")
User_tweets = pd.merge(Tweetsle, user, on='id', how='inner')

User_tweets['avg_retweet_count'] = pd.to_numeric(User_tweets['avg_retweet_count'],errors='coerce')
User_tweets['avg_favorite_count'] = pd.to_numeric(User_tweets['avg_favorite_count'],errors='coerce')

User_tweets.to_csv('/Users/sunqiaoyubing/Desktop/User_tweets.csv', index = False)
Tweets_pro = pd.read_csv('/Users/sunqiaoyubing/Desktop/Tweets_pro.csv',sep=";")

#1
plt.title('How much represented tweet is a reply', size=20, y=1.1)
sns.countplot(x = 'in_apply', data=Tweets_pro)

#2
plt.title('Contrast of Average in apply for Bots and Humans', size=20, y=1.1)
df1 = User_tweets[['avg_in_apply','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['avg_in_apply'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['avg_in_apply'], shade=True, color="b",label='Bots')

#3
plt.title('Contrast of Average text diversity for Bots and Humans', size=20, y=1.1)
df1 = User_tweets[['avg_ext_diversity','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['avg_ext_diversity'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['avg_ext_diversity'], shade=True, color="b",label='Bots')

#4
plt.title('Contrast of average retweet count for Bots and Humans', size=20, y=1.1)
df1 = User_tweets[['avg_retweet_count','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['avg_retweet_count'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['avg_retweet_count'], shade=True, color="b",label='Bots')

#5
plt.title('Contrast of average reply count for Bots and Humans', size=20, y=1.1)
df1 = User_tweets[['avg_reply_count','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['avg_reply_count'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['avg_reply_count'], shade=True, color="b",label='Bots')

#6
plt.title('Contrast of average favorite count for Bots and Humans', size=20, y=1.1)
df1 = User_tweets[['avg_favorite_count','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['avg_favorite_count'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['avg_favorite_count'], shade=True, color="b",label='Bots')

#7
plt.title('Contrast of average number of hashtags for Bots and Humans', size=20, y=1.1)
df1 = User_tweets[['avg_num_hashtags','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['avg_num_hashtags'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['avg_num_hashtags'], shade=True, color="b",label='Bots')

#8
plt.title('Contrast of average number of url for Bots and Humans', size=20, y=1.1)
df1 = User_tweets[['avg_num_urls','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['avg_num_urls'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['avg_num_urls'], shade=True, color="b",label='Bots')

#9
plt.title('Contrast of average number of mentions for Bots and Humans', size=20, y=1.1)
df1 = User_tweets[['avg_num_mentions','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['avg_num_mentions'], shade=True, color="r",label='Humans')
p1=sns.kdeplot(bot_len1['avg_num_mentions'], shade=True, color="b",label='Bots')

#10
X_2 = Tweetsle.drop(['id'], axis=1)
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(X_2.corr(), cmap='coolwarm', annot=True, linewidths=.5, fmt= '.1f',ax =ax,square=True)



##################################################################################################
## 2. User_network Network data visulization
user_network = pd.merge(network, user, on='id', how='inner')

user_network.to_csv('/Users/sunqiaoyubing/Desktop/user_network.csv', index = False)

#1
plt.title('Distribution of ff_ratio for Humans', size=20, y=1.1)
df1 = level3[['ff_ratio','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len0['ff_ratio'], shade=True, color="r",label='Humans')

#2
plt.title('Distribution of ff_ratio for Bots', size=20, y=1.1)
df1 = level3[['ff_ratio','Bot']]
bot_len1  = df1.ix[(df1['Bot']==1)]
bot_len0  = df1.ix[(df1['Bot']==0)]
p1=sns.kdeplot(bot_len1['ff_ratio'], shade=True, color="b",label='Bots')

#3
F_network = network.drop(['id'], axis=1)
f,ax = plt.subplots(figsize=(14, 10))
sns.heatmap(F_network.corr(), cmap='coolwarm', annot=True, linewidths=.5, fmt= '.1f',ax =ax)

##################################################################################################
## 
























