#!/usr/bin/env python
# coding: utf-8

# # V. Identifying Attributes of the Proper Names (PN): Professions and Roles

# In order to build a node list for all personal names (PN) mentioned in each text, we create a dictionary with all the PN as keys. The values are dictionaries that contain additional attributes for each PN, including their profession and role. These will be added to the resulting node list below. 
# 
# The goal of this notebook is to build a node list (as `CSV`) with each row representing a PN with a unique ID, and additional columns for the attributes of the PN. Currently, these include the associated role for each text and their profession if matched with the list below. 

# In[ ]:


people = dict()
for row in filtered.itertuples():
    if 'PN' in row.lemma:
        people[(row.lemma, row.id_word)] = dict()


# In[ ]:


#sanity check
print(people.keys())


# ## 1 List of Professions

# We first created the initial list of professions and other named entities by mining all the nouns in each tablet which follow each PN, and placing them in a `CSV` entitled [words_after_PN](https://docs.google.com/spreadsheets/d/1Jrn8nzMl59CTd8qdiwFZCuCbogj-U5FdBXhF26_94U0/edit?usp=sharing). 
# We then split the list based on different lexical boundaries and used a typology to categorize these into different groups: 
# * com = commodity
# * econ = economic terminology
# * fam = family affiliation
# * gender = term distinguishing gender
# * place = geographic names, house names, place names, etc.
# * prof = professions and titles used
# * time = chronological terms, or things associated with the cultic calendar (e.g. ‘offerings’)
# 
# SOURCE: [URAP meeting notes](https://docs.google.com/document/d/1GRyje1Qmt0tbi6PZHk0lJNvFpwE0mIsvIG08PAj4Q_c/edit?usp=sharing)

# In[ ]:



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
                # "urin[guard]", this is a verb
                "UN.IL₂[menial]",
                "ušbar[weaver]",
                "zabardab[official]",
                "zadim[stone-cutter]"]

profession_counts = { i : 0 for i in professions }


# If previous word is a 'PN' and the current word matches a profession, I'll add that profession to the PN. The commented out parts are from when I tried to generate my own list of professions. 

# In[ ]:


import re

prev_row = None
for row in filtered.itertuples():
        if prev_row != None and 'PN' in prev_row.lemma and (re.match('^[^\]]*', row.lemma)[0] + "]") in professions:
                profession_counts[(re.match('^[^\]]*', row.lemma)[0] + "]")] += 1
                people[(prev_row.lemma, prev_row.id_word)]['profession'] = row.lemma
        prev_row = row
        
        
english = set()

#for word in words:
    #to get only english use this one
    #english.append((re.search('\[(.*?)\]', str(word)).group()))
 #   english.add(word[0])

#print(len(english))

#for val in set(english):
 #   print(val[1:-1])

#Since this is related to the n-neighbors, this CSV is later on used in the neighbors notebook
pd.DataFrame(list(english)).to_csv('words_after_PN.csv', index = False)


# In[ ]:


pd.DataFrame.from_dict(profession_counts, orient='index', columns=['Percentages']).head()


# In[ ]:


total = sum(profession_counts.values())
profession_percentages = { i : profession_counts[i]/total for i in profession_counts}
percentage_df = pd.DataFrame.from_dict(profession_percentages, orient='index', columns=['Percentage'])
percentage_df.head(15)


# Looking at professions that never appear:

# In[ ]:


percentage_df[percentage_df['Percentage'] ==  0.0]


# In[ ]:


percentage_df.sort_values(by='Percentage', ascending=False)


# ## 2 Finding Roles

# More information on keywords can be found here:
# https://github.com/niekveldhuis/Sumerian-network/blob/master/Finding%20Merging%20Roles%20and%20Professions/roles_and_keywords.md 

# In[ ]:


import re
def match_ki_ta():
    prev_row = None
    i=0
    words = []
    for row in filtered.itertuples():
        if re.match(r".+ta\[\]PN", row.lemma) and 'ki[place]N' in prev_row:
            words.append((row.lemma, row.id_word))
            i+=1
        prev_row = row
    print('Number of roles found:', i)
    return (words, i)


def find_keyword_prev(keyword):
    prev_row = None
    i=0
    words = []
    for row in filtered.itertuples():
        if prev_row != None and keyword in row.lemma and 'PN' in prev_row.lemma:
            words.append((prev_row.lemma, prev_row.id_word))
            i+=1
        prev_row = row
    print('Number of roles found:', i)
    return (words, i)
    
def find_keyword_next(keyword):
    prev_row = None
    i=0
    words = []
    for row in filtered.itertuples():
        if prev_row != None and keyword in prev_row.lemma and 'PN' in row.lemma:
            words.append((row.lemma, row.id_word))
            i+=1
        prev_row = row
    print('Number of roles found:', i)
    return (words, i)


# Creating a dictionary that maps roles to count of those roles

# In[ ]:


role_counts = dict()


# 1. Matching rows that have 'ki[place]N' in previous cell and '-ta' in current

# In[ ]:


#source
for person, id_w in match_ki_ta()[0]:
    try:
        people[(person, id_w)]['role'] = 'source'
    except:
        print(person, id_w)
        
role_counts['source'] = match_ki_ta()[1]


# 2. Matching rows that have 'dab[seize]V/t' in lemmatization

# In[ ]:


res = find_keyword_prev('dab[seize]V/t')
for (person, id_w) in res[0]:
    try:
        people[(person, id_w)]['role'] = 'recipient'
    except:
        print(person, id_w)
        
role_counts['recipient'] = res[1]


# 3. Matching rows that have 'mu.DU[delivery]N' in next cell

# In[ ]:


# new owner
res = find_keyword_next('mu.DU[delivery]N')
for (person, id_w) in res[0]:
    try:
        people[(person, id_w)]['role'] = 'new owner'
    except:
        print((person, id_w))
        
role_counts['new owner'] = res[1]


# 4. Matching rows that have 'šu[hand]N' in next cell

# In[ ]:


res = find_keyword_prev('šu[hand]N')
for (person, id_w) in res[0]:
    try:
        people[(person, id_w)]['role'] = 'recipient'
    except:
        print(person, id_w)
role_counts['recipient'] += res[1]


# 5. Matching rows that have 'ŋiri[foot]N' in next cell

# In[ ]:


res = find_keyword_next('ŋiri[foot]N')
for (person, id_w) in res[0]:
    try:
        people[(person, id_w)]['role'] = 'intermediary'
    except:
        print((person, id_w))
        
role_counts['intermediary'] = res[1]        


# 6. Looks for rows with 'maškim' in previous spot

# In[ ]:


res = find_keyword_prev('maškim[administrator]N')
for (person, id_w) in res[0]:
    try:
        people[(person, id_w)]['role'] = 'administrator'
    except:
        print((person, id_w))
        
role_counts['administrator'] = res[1]      


# 7. Looking for PNs before and after 'zig[rise]V/i'

# In[ ]:


res = find_keyword_next('zig[rise]V/i')
for (person, id_w) in res[0]:
    try:
        people[(person, id_w)]['role'] = 'source'
    except:
        print((person, id_w))
        
role_counts['source'] += res[1]


# In[ ]:


res = find_keyword_prev('zig[rise]V/i')
for (person, id_w) in res[0]:
    try:
        people[(person, id_w)]['role'] = 'source'
    except:
        print((person, id_w))
        
role_counts['source'] += res[1]


# This is every PN mentioned in the filtered texts

# In[ ]:


len(people.keys())


# 

# In[ ]:


people_df = pd.DataFrame(list(people.items()))
people_df.head(4)


# ## 3 Transfer to CSV

# In[ ]:


name_word = pd.DataFrame(people_df[0].values.tolist(), index=people_df.index, columns = ['Name', 'id_word'])
final = name_word.join(people_df.drop(people_df.columns[0], axis = 1))


# In[ ]:


final = pd.concat([final.drop([1], axis=1), final[1].apply(pd.Series)], axis=1)


# In[ ]:


final.head()


# adding CDLI No column

# In[ ]:


CDLI_No = final['id_word'].apply(lambda x: re.split('\.', x)[0])
final.insert(4, 'CDLI No', CDLI_No)
final.head()


# In[ ]:



final.to_csv('roles_professions.csv')


# In[ ]:


#TODO: need to deal with case when PN not right next to role
[(row.lemma, row.id_word) for row in filtered.itertuples() if 'P330639.17' in row.id_word]


# ## 4 Analyzing Roles and Professions

# Here we analyze percentages of roles. Previously we created a dataframe for percentages of professions

# In[ ]:


#role percentages

total = sum(role_counts.values())
role_counts.update((k,role_counts[k]/total) for k in role_counts)
print(role_counts)
pd.DataFrame.from_dict(role_counts, orient='index', columns=['Percentages'])

