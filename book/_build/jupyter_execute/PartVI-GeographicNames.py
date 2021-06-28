#!/usr/bin/env python
# coding: utf-8

# # VI. Geographic Names

# 

# In[ ]:


files = ['filtered_'+str(n)+'_with_neighbors_normalized.csv' for n in range(1,10)]


# In[ ]:


geo_names = {}
p_number = ''
places = []
ids = []
geo_names_list = []
types = ["SN", "GN", "TN", "AN", "FN", "QN", "WN"]
for index, row in words_df.head(500).iterrows():
  if row['id_word'][:7] != p_number:
      geo_names[p_number] = places
      p_number = row['id_word'][:7]
      places = []
    #  ids = []
  w = row['lemma']
  if w[-2:] in types:
      places.append(w)
      ids.append(row['id_word'])
     # print(ids)
      if w not in geo_names_list:
          geo_names_list.append(w)
geo_names[p_number] = [places]
del geo_names['']


# In[ ]:


d = {'ip':list(geo_names.keys()),'geo_name': list(geo_names.values())}
geo_df = pd.DataFrame.from_dict(data = d)
#geo_names_list


# In[ ]:


geo_df["num_geo_names"] = geo_df["geo_name"].apply(lambda x:len(x))


# In[ ]:


geo_df[geo_df.num_geo_names>1].head(10)

