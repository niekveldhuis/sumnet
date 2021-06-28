#!/usr/bin/env python
# coding: utf-8

# # II. N-gram Neighbors of the Proper Names (PN)

# Section II was made to add greater context to each PN.
# 
# Below are lists of professions, roles, and family relationships.

# In[1]:


# import necessary libraries
import pandas as pd
from tqdm.auto import tqdm

# import libraries for this section
import re


# ## 1 Find Neighbors
# 
# Below we are making a copy of the filtered dataframe to manipulate and add the neightbors column.
# 
# The commented out line can be used if you have a copy of the words_df dataframe from the previous section and you would like to load that instead of running part I.

# In[2]:


words_df = pd.read_csv('output/part_1_output.csv')
words_df = pd.read_pickle('output/part_1_output.p')


# In[3]:


#List of professions, roles, family
professions = [ "aʾigidu[worker]", 
                "abala[water-drawer]", 
                "abrig[functionary]", 
                "ad.KID[weaver]", 
                "agaʾus[soldier]",
                "arad[slave]",
                "ašgab[leatherworker]",
                "aʾua [musician]", 
                "azlag[fuller]",
                "bahar[potter]",
                "bisaŋdubak[archivist]",
                "damgar[merchant]",
                "dikud[judge]",
                "dubsar[scribe]",
                "en[priest]",
                "erešdiŋir[priestess]",
                "ensik[ruler]",
                "engar[farmer]",
                "enkud[tax-collector]",
                "gabaʾaš[courier]",
                "galamah[singer]",
                "gala[singer]",
                "geme[worker]",
                "gudug[priest]",
                "guzala[official]",
                "idu[doorkeeper]",
                "išib[priest]",
                "kaguruk[supervisor]",
                "kaš[runner]",
                "kiŋgia[messenger]",
                "kinda[barber]", 
                "kinkin[miller]",
                "kiridab[driver]", 
                "kurušda[fattener]", 
                "kuš[official]",
                "lu[person]",
                "lugal[king]",
                "lukur[priestess]",
                "lungak[brewer]",
                "malah[sailor]",
                "muhaldim[cook]",
                "mušendu[bird-catcher]",
                "nagada[herdsman]",
                "nagar[carpenter]",
                "nar[musician]",
                "nargal[musician]", 
                "narsa[musician]", 
                "nin[lady]",
                "nubanda[overseer]",
                "nukirik[horticulturalist]",
                "saŋ.DUN₃[recorder]",
                "saŋŋa[official]",
                "simug[smith]",
                "sipad[shepherd]",
                "sukkal[secretary]",
                "šabra[administrator]",
                "šagia[cup-bearer]",
                "šakkanak[general]",
                # "szej[cook]", this is a verb
                "šidim[builder]",
                "šuʾi[barber]",
                "šukud[fisherman]",
                "tibira[sculptor]",
                "ugula[overseer]",
                "unud[cowherd]",
                # "urin[guard]",
                "UN.IL₂[menial]",
                "ušbar[weaver]",
                "zabardab[official]",
                "zadim[stone-cutter]"]

roles = ['ki[source]', 'maškim[administrator]', 
         'maškim[authorized]', 'i3-dab5[recipient]', 'giri3[intermediary]']

family = ['šeš[brother]', 'szesz[brother]', 'dumu[son]', 'dumu-munus[daughter]', 
        'dumumunus[daughter]' , 'dam[spouse]']


# In[4]:


def n_neighbors(data, n):
    #create list to return, non-proper names will return empty lists
    n_neighbors_list = [[] for i in range(len(data))]

    #find list of all PN lemma indices
    PN_index = data[data['lemma'].str.contains("PN")].index

    #go through each tablet and find neighbors for each PN and add to list
    for i in tqdm(PN_index, desc='N Neighbors'):
        
        #find all lemma rows from the same tablet
        group_of_same_pnumber = data[data['id_text'] == data.loc[i, 'id_text']]

        #find all lemma rows from the n-gram range
        group_of_n_lines_befaf = group_of_same_pnumber[((group_of_same_pnumber['id_line'] >= data.loc[i, 'id_line'] - n)
                                                        &(group_of_same_pnumber['id_line'] <= data.loc[i, 'id_line']))
                                                    | ((group_of_same_pnumber['id_line'] <= data.loc[i, 'id_line'] + n)
                                                       & (group_of_same_pnumber['id_line'] >= data.loc[i, 'id_line']))]
        
        #create list of n-grams and remove breaks
        lemma_neighbors = group_of_n_lines_befaf['lemma'].values.tolist()
        if 'break' in lemma_neighbors:
            lemma_neighbors.remove('break')

        #add to final list
        n_neighbors_list[i] = lemma_neighbors

    return n_neighbors_list


# In[5]:


def test_n_neighbors(data, n):
   #create list to return, non-proper names will return empty lists
    n_neighbors_list = [[] for i in range(len(data))]
    
    sorted_rows = data.sort_values(by=['id_text', 'id_line']).iterrows()
    
    current_line = 1
    previous_text = ''
    last_n_lines = []
    for row_list in sorted_rows:
      row = row_list[1]
      if row['id_text'] != previous_text:
        last_n_lines = [[row]]
        current_line += 1
        break
        
      if row['id_line'] != previous_text:
        if len(last_n_lines) == n:
          last_n_lines[0:(n-2)] = last_n_lines[1:(n-1)]
          last_n_lines[n-1] = [row['lemma']]
        else:
          last_n_lines.append([row['lemma']])


# In[6]:


words_df['prof?'] = words_df['lemma'].apply(lambda word: 'Yes' if (re.match('^[^\]]*', word)[0] + ']') in professions else 'No')
words_df['role?'] = words_df['lemma'].apply(lambda word: 'Yes' if (re.match('^[^\]]*', word)[0] + ']') in roles else 'No')
words_df['family?'] = words_df['lemma'].apply(lambda word: 'Yes' if (re.match('^[^\]]*', word)[0] + ']') in family else 'No')

#Create "number?"" to see if row is number. this could imply that that next row is a commodity
words_df['number?'] = words_df['lemma'].str.contains('NU')
words_df['number?'] = ['Yes' if words_df['number?'][i] == True else 'No' for i in words_df.index]
words_df['commodity?'] = ['No'] + ['Yes' if words_df['number?'][i] == 'Yes' else 'No' for i in words_df.index[1:]]
words_df


# The next code block takes a very long time to run.

# In[7]:


#call n_neighbor function to get neighbors from two lines above and below
words_df['neighbors'] = n_neighbors(words_df, 2)
words_df


# Check output only has neighbors for proper names.

# In[8]:


words_df[words_df['lemma'].str.contains("PN")]


# In[9]:


words_df[~words_df['lemma'].str.contains("PN")]


# The following line confirms there are no rows where the lemma is not a Proper Noun and is given neighbors.

# In[10]:


sum([lst != [] for lst in words_df[~words_df['lemma'].str.contains("PN")]['neighbors']])


# ## 2 Save Results in CSV file & Pickle
# Here we will save the words_df output from parts 1 and 2.

# In[11]:


words_df.to_csv('output/part_2_output.csv')
words_df.to_pickle('output/part_2_output.p')

