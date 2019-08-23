#!/usr/bin/env python
# coding: utf-8

# BOT or NOT cresi-2015 dataset manipulation
## there are total 20 datasets to be merged, 5 users datasets and 5 tweets datasets.
## Besides 10 network datasets, 5 friends datasets and 5 followers datasets.

# Imports

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
#metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


#  1. User level Dataset Manipulating
## 1.1. concatenating datasets
# Import all users datasets and merged to one users dataset.
# There are total 5 datasets.


def get_train_data(in_file,arg):

    train_data = pd.DataFrame()

    for f in glob.glob(in_file):
        df = pd.read_csv(f, encoding='latin-1')
        train_data = train_data.append(df, ignore_index=True)

    train_data['Bot'] = arg

    return train_data


train1_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/E13.csv/users.csv",arg=0)
train2_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/FSF.csv/users.csv",arg=1)
train3_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/INT.csv/users.csv",arg=1)
train4_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/TFP.csv/users.csv",arg=0)
train5_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/TWT.csv/users.csv",arg=1)


user_data = pd.concat([train1_data,train2_data,train3_data,train4_data,train5_data])

user_data['default_profile'] = user_data['default_profile'].replace(np.nan, 0)
user_data['geo_enabled'] = user_data['geo_enabled'].replace(np.nan, 0)
user_data['profile_use_background_image'] = user_data['profile_use_background_image'].replace(np.nan, 0)
user_data['profile_background_tile'] = user_data['profile_background_tile'].replace(np.nan, 0)

# user_data['lang'].value_counts()
# user_data['location'].value_counts()
# user_data.head(2)

# variable "lang" analysis.
cleanup_nums = {"lang": {"en" :"English", "it":"Italian", "es":"Espanol", "pt": "Others",
                "ru": "Others", "fr": "Others", "tr": "Others","ko": "Others","de": "Others", "id": "Others", 
                "ar": "Others", "zh-cn": "Others", "gl": "Others", "nl": "Others", "ja": "Others"}}

user_data.replace(cleanup_nums, inplace=True)


plt.title('Contrast Bot and Humas with different languages', size=20, y=1.1)
sns.countplot(x = 'lang', hue='Bot', data=user_data)

## 1.2 User level data transfromation
user_data.loc[:,'description'].fillna("")    
user_data.loc[:,'sn_length'] = user_data.screen_name.apply(lambda text: len(str(text)))
user_data.loc[:,'desc_length'] = user_data.description.apply(lambda text: len(str(text)))
grouped = user_data['screen_name']
user_data.loc[:,"name_length"] = user_data.name.apply(lambda x: len(''.join(str(x))))   
#user_data.loc[:,'ff_ratio'] = (user_data.followers_count+1/user_data.friends_count+1)

user = user_data.drop(['url','location','lang','time_zone','screen_name','description','updated','name','default_profile_image','profile_image_url','profile_banner_url','created_at','dataset','profile_background_image_url_https','profile_image_url_https','utc_offset','profile_background_image_url','profile_use_background_image','protected','verified','profile_text_color','profile_sidebar_border_color','profile_link_color','ff_ratio','profile_sidebar_fill_color','profile_background_color','profile_use_background_image'], axis=1)


# user.head()
# user['profile_use_background_image'].value_counts()
# user['geo_enabled'].value_counts()
# user['default_profile'].value_counts()
# user['Bot'].value_counts()
# user['profile_background_tile'].value_counts()
# user.info()

## 1.3 User level data split and save
Train_cresi, Cresi_2 = train_test_split(user, test_size=0.4, random_state=25)
Test_cresi,Valid_cresi = train_test_split(Cresi_2, test_size=0.5, random_state=25)
# Train_cresi.shape, Valid_cresi.shape, Test_cresi.shape

user.to_csv('/Users/sunqiaoyubing/Desktop/user.csv', index = False)
Train_cresi.to_csv('/Users/sunqiaoyubing/Desktop/Train_cresi.csv', index = False)
Valid_cresi.to_csv('/Users/sunqiaoyubing/Desktop/Valid_cresi.csv', index = False)
Test_cresi.to_csv('/Users/sunqiaoyubing/Desktop/Test_cresi.csv', index = False)



#################################################################################
# 2. Tweets level datasets Manipulating
## 2.1 cancatenating and cleaning datasets
def get_tweets_data(in_file):
    tweets_data = pd.DataFrame()

    for f in glob.glob(in_file):
        df = pd.read_csv(f, encoding='latin-1')
        tweets_data = tweets_data.append(df, ignore_index=True)

    return tweets_data

tweets1_data = get_tweets_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/E13.csv/tweets.csv")
tweets2_data = get_tweets_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/FSF.csv/tweets.csv")
tweets3_data = get_tweets_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/INT.csv/tweets.csv")
tweets4_data = get_tweets_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/TFP.csv/tweets.csv")
tweets5_data = get_tweets_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/TWT.csv/tweets.csv")


tweets = pd.concat([tweets1_data,tweets2_data,tweets3_data,tweets4_data,tweets5_data])


# tweets
tweets.loc[:,'reply_count'] = tweets['reply_count'].fillna(0)
tweets.loc[:,'favorite_count'] = tweets['favorite_count'].fillna(0)
tweets.loc[:,'in_reply_to_screen_name'] = tweets['in_reply_to_screen_name'].fillna(0)
tweets.loc[:,'in_apply'] = tweets['in_reply_to_screen_name'].apply(lambda x: 1 if x is not 0 else 0)


tweets.reply_count = tweets.reply_count.apply(lambda x: int(x))
tweets.favorite_count = tweets.favorite_count.apply(lambda x: int(x))

# Tweets_pro
Tweets_pro = tweets.drop(['id','source','truncated','in_reply_to_user_id','in_reply_to_status_id','retweeted_status_id','geo','place','timestamp','in_reply_to_screen_name'], axis=1)

Tweets_pro.to_csv('/Users/sunqiaoyubing/Desktop/Tweets_pro.csv', index = False)

# Tweets_pro.info()


## 2.2. Transform tweets Features and made every accounts just has one line.

def lexical_diversity(text):
    if (len(text) == 0):
        diversity = 0
    else:
        diversity = float(len(set(text))) / len(text)

    return diversity


def aver_list(fea_list):

    # calculating average value
    return np.asarray(fea_list).mean()


def post_int_feature(fea_raw=""):

    #if the feature is null，then return zero：
     if len(fea_raw.strip())==0:
        fea_raw=0
    else:
        fea_raw=int(fea_raw.strip())

    return fea_raw



def post_float_feature(fea_raw=""):

    #if the feature is null，then return zero：

    if len(fea_raw.strip())==0:
        fea_raw=0
    else:
        fea_raw=float(fea_raw.strip())

    return fea_raw

def deal_with_infos_by_uerid(input_file,output_file):

    line_num=0
    user_id_infos_dict={}
    with open(input_file,'r')as fr,open(output_file,'w+')as fw:
        
        fw.write('id'+'\t'+'avg_ext_diversity'+'\t'+'avg_retweet_count'+'\t'+'avg_reply_count'+'\t'+'avg_favorite_count'+\
                 '\t'+'avg_num_hashtags'+'\t'+'avg_num_urls'+'\t'+'avg_num_mentions'+'\t'+'avg_in_apply'+'\n')

        for line in fr:

            line_num+=1
            if line_num==1:
                continue  # filter the first sample.
            line_split=line.rstrip('\n').split('\t')
            if len(line_split)!=9:
                # print("error line : ",len(line_split),line.rstrip('\n'))
                continue

            try:
                text_diversity=lexical_diversity(line_split[0].strip().split())

                user_id=line_split[1]
                if not  user_id.isdigit():
                    continue
                retweet_count=post_int_feature(line_split[2])
                reply_count =post_int_feature(line_split[3])
                favorite_count =post_int_feature(line_split[4])
                num_hashtags =post_int_feature(line_split[5])
                num_urls =post_int_feature(line_split[6])
                num_mentions =post_int_feature(line_split[7])
                in_apply=post_int_feature(line_split[8])

                if user_id not  in user_id_infos_dict:
                    user_id_infos_dict[user_id]=[[],[],[],[],[],[],[],[]]

                user_id_infos_dict[user_id][0].append(text_diversity)
                user_id_infos_dict[user_id][1].append(retweet_count)
                user_id_infos_dict[user_id][2].append(reply_count)
                user_id_infos_dict[user_id][3].append(favorite_count)
                user_id_infos_dict[user_id][4].append(num_hashtags)
                user_id_infos_dict[user_id][5].append(num_urls)
                user_id_infos_dict[user_id][6].append(num_mentions)
                user_id_infos_dict[user_id][7].append(in_apply)
            except:
                continue



        for k,infos in user_id_infos_dict.items():
            aver_fea=[]
            for i in range(8):
                aver_fea.append('%.3f'%(aver_list(infos[i])))
            fw.write(k+'\t'+'\t'.join(aver_fea)+'\n')
deal_with_infos_by_uerid(input_file='/Users/sunqiaoyubing/Desktop/Tweets_pro.txt',output_file='Tweetsle.txt')

#################################################################################
# 3. Network level datasets Manipulating
## 3.1 Friends datset
#cancatenating friends datasets
friends1_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/E13.csv/friends.csv",arg=0)
friends2_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/FSF.csv/friends.csv",arg=1)
friends3_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/INT.csv/friends.csv",arg=1)
friends4_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/TFP.csv/friends.csv",arg=0)
friends5_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/TWT.csv/friends.csv",arg=1)

friends_data = pd.concat([friends1_data,friends2_data,friends3_data,friends4_data,friends5_data])

#merge friends datsets with users datasets
friends_data['id'] = friends_data['target_id']
friends_data = friends_data.drop(['target_id','Bot'],axis=1)
#friends_data.head()

friend = pd.merge(friends_data, user_data, on='id', how='inner')
friend = friend.drop(['id'], axis=1)
#friend.head()

## 3.2. Transform friends Features and made every accounts just has one line.
def deal_with_friendT(input_file,output_file):

    line_num = 0
    user_id_infos_dict = {}
    with open(input_file, 'r')as fr, open(output_file, 'w+')as fw:
        
        
        fw.write('id'+'\t'+'avg_friends_statuses_count'+'\t'+'avg_friends_followers_count'+'\t'+
                 'avg_friends_friends_count'+'\t'+'avg_friends_favorite_count'+'\t'+'avg_friends_listed_count'+'\t'+
                 'avg_friends_default_profile'+'\t'+'avg_friends_geo_enabled'+'\t'+'avg_friends_profile_background_tile'+
        '\t'+'friend_Bot_rate'+'\t'+'avg_friends_sn_length'+'\t'+'avg_friends_desc_length'+'\t'+'avg_friends_name_length'+'\n')

        for line in fr:

            line_num += 1
            if line_num == 1:
                continue  # filter the frist sample
            line_split = line.rstrip('\n').split(',')
            if len(line_split) != 13:
                print("error line : ", len(line_split), line.rstrip('\n'))
                continue



            try:

                source_id = line_split[0]
                statuses_count = post_float_feature(line_split[1])
                followers_count = post_float_feature(line_split[2])
                friends_count = post_float_feature(line_split[3])
                favourites_count = post_float_feature(line_split[4])
                listed_count = post_float_feature(line_split[5])
                default_profile = post_float_feature(line_split[6])
                geo_enabled = post_float_feature(line_split[7])
                profile_background_tile = post_float_feature(line_split[8])
                Bot = post_float_feature(line_split[9])
                sn_length = post_float_feature(line_split[10])
                desc_length = post_float_feature(line_split[11])
                name_length = post_float_feature(line_split[12])




                if source_id not in user_id_infos_dict:
                    user_id_infos_dict[source_id] = [[],[],[],[],[],[],[],[],[],[],[],[]]



                user_id_infos_dict[source_id][0].append(statuses_count)
                user_id_infos_dict[source_id][1].append(followers_count)
                user_id_infos_dict[source_id][2].append(friends_count)
                user_id_infos_dict[source_id][3].append(favourites_count)
                user_id_infos_dict[source_id][4].append(listed_count)
                user_id_infos_dict[source_id][5].append(default_profile)
                user_id_infos_dict[source_id][6].append(geo_enabled)
                user_id_infos_dict[source_id][7].append(profile_background_tile)
                user_id_infos_dict[source_id][8].append(Bot)
                user_id_infos_dict[source_id][9].append(sn_length)
                user_id_infos_dict[source_id][10].append(desc_length)
                user_id_infos_dict[source_id][11].append(name_length)


            except:
                continue

        for k, infos in user_id_infos_dict.items():

            aver_fea = []
            for i in range(12):
                aver_fea.append('%.3f' % (aver_list(infos[i])))
            fw.write(k + '\t' + '\t'.join(aver_fea) + '\n')

deal_with_friendT(input_file='/Users/sunqiaoyubing/Desktop/friend.csv',output_file="friend_output.txt")

##### 3.3. Followers dataset
#cancatenating followers datasets
followers1_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/E13.csv/followers.csv",arg=0)
followers2_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/FSF.csv/followers.csv",arg=1)
followers3_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/INT.csv/followers.csv",arg=1)
followers4_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/TFP.csv/followers.csv",arg=0)
followers5_data=get_train_data("/Users/sunqiaoyubing/Desktop/thesis paper/cresci-2015.csv/TWT.csv/followers.csv",arg=1)


followers_data = pd.concat([followers1_data,followers2_data,followers3_data,followers4_data,followers5_data])

#merge followers datsets with users datasets
followers_data['id'] = followers_data['target_id']
followers_data = followers_data.drop(['target_id','Bot'],axis=1)
followers_data.head()

follower = pd.merge(followers_data, user_data, on='id', how='inner')
follower = follower.drop(['id'], axis=1)
follower.to_csv('/Users/sunqiaoyubing/Desktop/follower.csv', index = False)

## 3.4. Transform followers Features and made every accounts just has one line.
def deal_with_followersT(input_file,output_file):

    line_num = 0
    user_id_infos_dict = {}
    with open(input_file, 'r')as fr, open(output_file, 'w+')as fw:
        
        
        fw.write('id'+'\t'+'avg_followers_statuses_count'+'\t'+'avg_followers_followers_count'+'\t'+
                 'avg_followers_friends_count'+'\t'+'avg_followers_favorite_count'+'\t'+'avg_followers_listed_count'+'\t'+
                 'avg_followers_default_profile'+'\t'+'avg_followers_geo_enabled'+'\t'+'avg_followers_profile_background_tile'+
                 '\t'+'followers_Bot_rate'+'\t'+'avg_followers_sn_length'+'\t'+'avg_followers_desc_length'+'\t'
                  +'avg_followers_name_length'+'\n')

        for line in fr:

            line_num += 1
            if line_num == 1:
                continue  # 过滤第一条样本
            line_split = line.rstrip('\n').split(',')
            if len(line_split) != 13:
                print("error line : ", len(line_split), line.rstrip('\n'))
                continue



            try:

                source_id = line_split[0]
                statuses_count = post_float_feature(line_split[1])
                followers_count = post_float_feature(line_split[2])
                friends_count = post_float_feature(line_split[3])
                favourites_count = post_float_feature(line_split[4])
                listed_count = post_float_feature(line_split[5])
                default_profile = post_float_feature(line_split[6])
                geo_enabled = post_float_feature(line_split[7])
                profile_background_tile = post_float_feature(line_split[8])
                Bot = post_float_feature(line_split[9])
                sn_length = post_float_feature(line_split[10])
                desc_length = post_float_feature(line_split[11])
                name_length = post_float_feature(line_split[12])




                if source_id not in user_id_infos_dict:
                    user_id_infos_dict[source_id] = [[],[],[],[],[],[],[],[],[],[],[],[]]



                user_id_infos_dict[source_id][0].append(statuses_count)
                user_id_infos_dict[source_id][1].append(followers_count)
                user_id_infos_dict[source_id][2].append(friends_count)
                user_id_infos_dict[source_id][3].append(favourites_count)
                user_id_infos_dict[source_id][4].append(listed_count)
                user_id_infos_dict[source_id][5].append(default_profile)
                user_id_infos_dict[source_id][6].append(geo_enabled)
                user_id_infos_dict[source_id][7].append(profile_background_tile)
                user_id_infos_dict[source_id][8].append(Bot)
                user_id_infos_dict[source_id][9].append(sn_length)
                user_id_infos_dict[source_id][10].append(desc_length)
                user_id_infos_dict[source_id][11].append(name_length)


            except:
                continue

        for k, infos in user_id_infos_dict.items():

            aver_fea = []
            for i in range(12):
                aver_fea.append('%.3f' % (aver_list(infos[i])))
            fw.write(k + '\t' + '\t'.join(aver_fea) + '\n')

deal_with_followersT(input_file='/Users/sunqiaoyubing/Desktop/follower.csv',output_file="follower_output.txt")

#Merge friends and followers output datasets
#follower_output.info()
#friend_output.info()
follower_output=pd.read_csv('/Users/sunqiaoyubing/Desktop/follower_output.csv', encoding='latin-1',sep=";")
friend_output=pd.read_csv('/Users/sunqiaoyubing/Desktop/friend_output.csv', encoding='latin-1',sep=";")
network = pd.merge(friend_output, follower_output, on='id', how='inner')

network['avg_friends_statuses_count']=pd.to_numeric(network['avg_friends_statuses_count'],errors='coerce')
network['avg_friends_followers_count']=pd.to_numeric(network['avg_friends_followers_count'],errors='coerce')
network['avg_friends_friends_count']=pd.to_numeric(network['avg_friends_friends_count'],errors='coerce')
network['avg_friends_favorite_count']=pd.to_numeric(network['avg_friends_favorite_count'],errors='coerce')
network['avg_followers_statuses_count']=pd.to_numeric(network['avg_followers_statuses_count'],errors='coerce')
network['avg_followers_followers_count']=pd.to_numeric(network['avg_followers_followers_count'],errors='coerce')
network['avg_followers_friends_count']=pd.to_numeric(network['avg_followers_friends_count'],errors='coerce')
network['avg_followers_favorite_count']=pd.to_numeric(network['avg_followers_favorite_count'],errors='coerce')

## final network dataset
friend_output.loc[:,'ff_ratio'] = (user_data.followers_count+1/user_data.friends_count+1)

friend_output['avg_friends_statuses_count']=pd.to_numeric(friend_output['avg_friends_statuses_count'],errors='coerce')
friend_output['avg_friends_followers_count']=pd.to_numeric(friend_output['avg_friends_followers_count'],errors='coerce')
friend_output['avg_friends_friends_count']=pd.to_numeric(friend_output['avg_friends_friends_count'],errors='coerce')
friend_output['avg_friends_favorite_count']=pd.to_numeric(friend_output['avg_friends_favorite_count'],errors='coerce')
#friend_output.info()

network = friend_output.drop(['avg_friends_statuses_count','avg_friends_followers_count',
                              'avg_friends_friends_count','avg_friends_favorite_count'], axis=1)

network.to_csv('/Users/sunqiaoyubing/Desktop/network.csv', index = False)

###########################################################################################################
# All four level datasets
# Level 1:user level
user = pd.read_csv("/Users/sunqiaoyubing/Desktop/user.csv", encoding='latin-1')

# level 2: user+tweets level
Tweetsle = pd.read_csv("/Users/sunqiaoyubing/Desktop/Tweetsle.csv", encoding='latin-1')
User_tweets = pd.merge(Tweetsle, user, on='id', how='inner')
User_tweets['avg_retweet_count'] = pd.to_numeric(User_tweets['avg_retweet_count'],errors='coerce')
User_tweets['avg_favorite_count'] = pd.to_numeric(User_tweets['avg_favorite_count'],errors='coerce')

User_tweets.to_csv('/Users/sunqiaoyubing/Desktop/User_tweets.csv', index = False)

# level 3: user+network level
user_network = pd.merge(network, user, on='id', how='inner')
user_network.loc[:,'ff_ratio'] = (user_network.followers_count+1/user_network.friends_count+1)
user_network.to_csv('/Users/sunqiaoyubing/Desktop/user_network.csv', index = False)

#delete inf
with open('/Users/sunqiaoyubing/Desktop/User_network.csv')as fr,open('new_User_network.csv','w+')as fw:


    for line in fr.readlines():

        line_split=line.rstrip('\n').split(',')
        if len(line_split)!=22:
            print('error line: ',len(line_split),line.rstrip('\n'))

        if 'inf' in line_split[-13]:
            continue

        fw.write(line)

# level 4: user+tweets+network level
All = pd.merge(network, User_tweets, on='id', how='inner')
All['avg_retweet_count'] = All['avg_retweet_count'].replace(np.nan, 0)
All.info()
User_network= pd.read_csv('/Users/sunqiaoyubing/Desktop/new_User_network.csv', encoding='latin-1')

All.to_csv('/Users/sunqiaoyubing/Desktop/All.csv', index = False)

with open('/Users/sunqiaoyubing/Desktop/All.csv')as fr,open('new_All.csv','w+')as fw:


    for line in fr.readlines():

        line_split=line.rstrip('\n').split(',')
        if len(line_split)!=22:
            print('error line: ',len(line_split),line.rstrip('\n'))

        if 'inf' in line_split[-13]:
            continue

        fw.write(line)

























