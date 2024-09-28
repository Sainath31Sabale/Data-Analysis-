#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on IPL matches dataset

# In[1]:


import pandas as pd
#data preprocessing

import numpy as np
#numerical python mathematical calculations

import matplotlib.pyplot as plt
#visualization library

import seaborn as sns 
#visuallization library


# In[3]:


df = pd.read_csv("IPL_matches_dataset.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# ### A)Pre-Match Analysis 
# <pre>
# <b>1) Number of matches played in season</b>
#    Most matches were played in IPL season 2013 <br>
# <b>2) Team winning most tosses</b>
#    Mumbai Indian has won most number of tosses<br>
# <b>3) Toss Decision(Bat or to Field) in each IPL</b>
#    There is a trend in the IPL seasons 2016 onwards, the decision to field  first has been dominant.
# <b>4) Toss Decision Percentage</b>
# 
# <b>5) Most frequently used stadiums</b>
#    Most matches have been played in Eden Gardens stadium, which is followed by M Chinnaswamy Stadium<br>
# <b>6) City that hosted most IPL matches across all seasons<\b>
#    Most number of IPL matches across all seasons have been played in Mumbai followed by Kolkata<br>
# <b>7) Top 5 Umpire1 who evaluate diffrent matches across all IPL seasons</b>
#    Top 5 Umpire1 who evaluate diffrent matches across all IPL seasons
#    
# </pre>

# # Most number of matches played in season

# In[7]:


df["Season"].value_counts()


# In[13]:


plt.figure(figsize=(10,5))
ax = sns.countplot(x="Season",data=df,
                  order = df["Season"].value_counts().index.sort_values(),palette = "Set1")
for p in ax.patches:
    ax.annotate("{}".format(p.get_height()),(p.get_x()+0.25,p.get_height()+1))
plt.title("Matches played in each IPL season",fontsize=20)
plt.xlabel("IPL Season",fontsize=20)
plt.ylabel("Count")
plt.show()


# #### Inference 
# 
# 1) Most matches were played in IPL season 2013

# # Toss Winner(Team winning most tosses)

# In[15]:


plt.figure(figsize=(10,5))
sns.countplot(y="toss_winner",data=df,
             order = df["toss_winner"].value_counts().index,
             palette="magma")
plt.ylabel("Toss winning team",fontsize=13)
plt.xlabel("Count",fontsize=13)
plt.title("Teams who won most tosses",fontsize=17)
plt.show()


# #### Inference
# 
# Mumbai Indian has won most number of tosses

# # Toss Decision(Bat or to Field) in each IPL

# In[16]:


plt.figure(figsize=(15,7))
sns.countplot(x="Season",hue="toss_decision",data=df,
             order=df["Season"].value_counts().index.sort_values(),
             palette="Set1")
plt.title("Decision to field or Bat in each IPL season")
plt.xlabel("Season",fontsize=13)
plt.ylabel("Count",fontsize=13)
plt.show()


# #### Inference
# 
# 1) There is a trend in the IPL seasons 2016 onwards, the decision to field first has been dominant
# 
# 2) Decision to field is more dominant across all IPL seasons except in IPL 2009,IPL 2012,IPL 2013.

# # Toss Decision Percentage

# In[19]:


df['toss_decision'].value_counts()


# In[21]:


r1 = df["toss_decision"].value_counts()
r1


# In[23]:


plt.figure(figsize=(8,6))
plt.pie(x=r1.values,labels=r1.index,autopct="%.2f%%",explode=[0,0.1],colors=["red","blue"])
plt.title("Toss decision percentage")
plt.show()


# # Most frequently used stadiums

# In[26]:


plt.figure(figsize=(10,6))
sns.countplot(y="venue",data=df,order=df["venue"].value_counts()[:10].index,palette="bright")
plt.xlabel("Count",fontsize=13)
plt.ylabel("Stadium name",fontsize=13)
plt.title("Number of matches played at each stadium",fontsize=17)
plt.show()


# #### Inference 
# Most matches have been played in Eden Gardens stadium, which is followed by M Chinnaswamy Stadium

# # City that hosted most IPL matches across all seasons

# In[28]:


df.columns


# In[30]:


plt.figure(figsize=(10,6))
sns.countplot(y="city",data=df,order=df["city"].value_counts()[:7].index,palette="bright")
plt.xlabel("Count",fontsize=13)
plt.ylabel("city name",fontsize=13)
plt.title("City that hosted monst number of matches across all seasons",fontsize=17)
plt.show()


# #### Inference
# 
# 1) Most number of IPL matches across all seasons have been played in Mumbai followed by Kolkata

# # Top 5 Umpire1 who evaluate diffrent matches across all IPL seasons

# In[33]:


df.columns


# In[35]:


plt.figure(figsize=(10,6))
sns.countplot(y="umpire1",data=df,order=df["umpire1"].value_counts()[:5].index,palette="bright")
plt.xlabel("Count",fontsize=13)
plt.ylabel("Umpire1 Names",fontsize=13)
plt.title("Top5 Umpire1 who were to evaluate IPL matches across all seasons",fontsize=17)
plt.show()


# #### Inference
# 
# 1) HDPK Dharmasena has evaluated most IPL matches

# # Post Match Analysis
# <pre>
# 1)Team that won most of the matches across all the IPL Seaseons
# 2)Percentage distribution of different winning teams(Top 8) across all the IPL seasons
# 3)Top 7 players with most player of the match awards
# 4)Win Toss to game
# 5)Win by Runs(How many Times each team has won the match by runs)
# 6)Win by wickets(How many times each IPL team has won the match by wickets)
# </pre>

# #### 1) Team that won most of the matches across all the IPL Seaseons

# In[40]:


df.columns


# In[41]:


plt.figure(figsize=(10,6))
sns.countplot(y="winner",data=df,order=df["winner"].value_counts().index,palette="magma")
plt.xlabel("Count",fontsize=13)
plt.xlabel("Winning Team",fontsize=13)
plt.title("Team that won the most of the matches across all the IPL Seasons",fontsize=16)
plt.show()


# #### Inference
# 1) Mumbai Indians is the team that has won the most of the matches across all the IPL seasons, followed by chennai super kings

# ### 2) Percentage distribution of different winning teams(Top 8) across all the IPL seasons

# In[42]:


df['winner'].value_counts()


# In[47]:


df['winner'].value_counts()[:8].plot(kind='pie',
                                    title='Win percentage across all IPL seasons',cmap='RdBu',
                                    autopct='%2.f%%',figsize=(7,6))
plt.show()


# #### Inference
# 1) Mumbai Indians has won 16% of all the IPL matches across all the seaseons, followed by chennai super kings at 15%

# ### 3) Top 7 players with most player of the match awards

# In[49]:


r1 = df['player_of_match'].value_counts()[:7]
r1


# In[54]:


plt.figure(figsize=(8,6))
ax = sns.barplot(x=r1.index,y=r1.values,palette='viridis')
for p in ax.patches:
    ax.annotate('{}'.format(int(p.get_height())),(p.get_x()+0.25,p.get_height()+0.4))
plt.title('Most player of the match won',fontsize=17)
plt.xlabel("Player",fontsize=13)
plt.ylabel("Count",fontsize=13)
plt.show()


# #### Inference
# 1) Chris Gayle has won the most player of the match awards,followed by AB Devilliers

# ### 4) Win Toss to game

# In[57]:


df['win_toss_to_game']=(df['toss_winner']==df['winner'])
df[['team1','team2','toss_winner','winner','win_toss_to_game']].head()


# In[58]:


df['win_toss_to_game']=np.where(df['win_toss_to_game']==True,'Win','Lose')


# In[59]:


r2=df['win_toss_to_game'].value_counts()
r2


# In[60]:


plt.bar(r2.index,r2.values,color=['green','red'])
plt.title('Win Toss to Game',fontsize=17)
plt.xlabel("Win or Lose",fontsize=13)
plt.ylabel("Count",fontsize=13)
plt.show()


# #### Inference
# 1) Across all the IPL seasons in 393 matches, the toss winner emerged as the match winner as well as and in 363 matches the toss winner lost the match

# ### 5) Win by Runs(How many Times each team has won the match by runs)

# In[61]:


df.head()


# In[63]:


cont_win_by_runs_per_team={}
for i in df['team1'].unique():
    win_team = df[df['winner']==i]
    print('Team',i)
    cont_win_by_runs_per_team[i]= win_team[win_team['win_by_runs']!=0].shape[0]


# In[64]:


print(cont_win_by_runs_per_team)


# In[70]:


plt.figure(figsize=(9,7))
plt.barh(list(cont_win_by_runs_per_team.keys()),list(cont_win_by_runs_per_team.values()))
plt.title('IPL team wise total win by Runs',fontsize=17)
plt.xlabel("Count of win by runs",fontsize=13)
plt.ylabel("IPL Teams",fontsize=13)
plt.grid()
plt.show()


# #### Inference
# 1) Mumbai Indians has won most of the matches by runs

# ### 6) Win by wickets(How many times each IPL team has won the match by wickets)

# In[72]:


count_win_by_wickets_per_team={}
for i in df['team1'].unique():
    win_team = df[df['winner']==i]
    print('Team',i)
    count_win_by_wickets_per_team[i]= win_team[win_team['win_by_wickets']!=0].shape[0]


# In[73]:


print(count_win_by_wickets_per_team)


# In[75]:


plt.figure(figsize=(9,7))
plt.barh(list(count_win_by_wickets_per_team.keys()),list(count_win_by_wickets_per_team.values()))
plt.title('IPL team wise total win by Wickets',fontsize=17)
plt.xlabel("Count of win by wickets",fontsize=13)
plt.ylabel("IPL Teams",fontsize=13)
plt.grid()
plt.show()


# #### Inference
# 1) Kolkata Knight Riders have won most of the IPL matches based on win by wickets, followed by Mumbai Indians and Chennai Super Kings

# # C) Team Wise Analysis

# ### 1) Mumbai Indians
# 

# In[76]:


mi = df[(df['team1']=='Mumbai Indians') | (df['team2']=='Mumbai Indians')]
mi.shape


# In[77]:


mi_win_loss = mi['winner'].value_counts()
mi_win_loss


# In[83]:


plt.barh(width=mi_win_loss.values,y=mi_win_loss.index,color='orange',edgecolor='black')
plt.title("Mi win performance over other teams")
plt.ylabel("Teams")
plt.xlabel("Number of matches won")
plt.grid()
plt.show()


# #### Inference
# 1) MI has won 109 times when compared to the losses against it

# ### 2) Chennai Super Kings

# In[84]:


csk = df[(df['team1']=='Chennai Super Kings') | (df['team2']=='Chennai Super Kings')]
csk.shape


# In[85]:


csk_win_loss = csk['winner'].value_counts()
csk_win_loss


# In[86]:


plt.barh(width=csk_win_loss.values,y=csk_win_loss.index,color='orange',edgecolor='black')
plt.title("CSK win performance over other teams")
plt.ylabel("Teams")
plt.xlabel("Number of matches won")
plt.grid()
plt.show()


# #### Inference
# 1) CSK has won 100 times when compared to the losses against it
