#!/usr/bin/env python
# coding: utf-8

# # VI. Geography

# In this section, we make use of the [Geographic Data for Ancient Near Eastern Archaeological Sites](https://www.lingfil.uu.se/research/assyriology/earth/) from Professor [Olof Pedersén](https://katalog.uu.se/profile/?id=N94-579) at Uppsala University to locate some of the geographic names appeared in the Drehem texts. 
# 
# We will use fuzzy search to find matches between the ANE files and our texts, and then use the geographic locations in the ANE to plot those that appear in our texts.

# The following installs and imports the relevant libararies.

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().run_line_magic('cd', '"drive/My Drive/SumerianNetworks"')


# In[ ]:


get_ipython().system('pip install geopandas')
get_ipython().system('pip install descartes')
get_ipython().system('pip install pysal')
get_ipython().system('pip install fuzzywuzzy')
get_ipython().system('pip install python-Levenshtein')
get_ipython().system('pip install NetworkX')


# In[ ]:


import pandas as pd
import numpy as np
import geopandas as gpd
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 
import fiona
fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by defaults
import networkx as nx


# We first import filtered Drehem data and extract all the geographic names: SN (settlement name), TN (temple name), AN, FN (field name), QN (quarter name), WN (water course name).

# In[ ]:


drehem = pd.read_csv("JupyterBook/words_df.csv")
drehem.head()


# In[ ]:


drehem.shape


# In[ ]:


geo_names = {}
p_number = ''
places = []
geo_names_list = []

#SN (settlement name), TN (temple name), AN, FN (field name), QN (quarter name), WN (water course name)
#remove year names
types = ["SN", "TN", "AN", "FN", "QN", "WN"]
for index, row in drehem.iterrows():
      # if row['id_word'][:7] != p_number:
      #     geo_names[p_number] = places
      #     p_number = row['id_word'][:7]
      #     places = []
      w = row['lemma']
      if w[-2:] in types:
          geo_names[row['id_text']] = w
# geo_names[p_number] = [places]
# del geo_names['']


# In[ ]:


d = {'ip':list(geo_names.keys()),'geo_name': list(geo_names.values())}
geo_df = pd.DataFrame.from_dict(data = d)
geo_df.head()


# Now we remove the suffixes and leave just the geographical names. (eg. "[00]SN").

# In[ ]:


def remove_tail(string):
  return string.split("[")[0]

non_emp_geo = geo_df
non_emp_geo["cleaned_geo_name"] = non_emp_geo["geo_name"].map(remove_tail)
non_emp_geo.head()


# We now import the ANE KMZ files. Make sure you have the latest files from [here](https://www.lingfil.uu.se/research/assyriology/earth/) and change the file name below accordingly.

# In[ ]:


location = gpd.read_file('JupyterBook/jupyter_collection/04_Geographic_Names/geopandas_intro/data/sites.kml', driver='KML',layer=2)
print(location.shape)
location.head()


# Since the `Name` column contains both the modern name and ancient name of the same place, we will convert them into a list for matching purpose.

# In[ ]:


def parse_names(string):
    """Takes in a string (eg. Adamdun? (Teppe Surkhehgan))
    Returns a tuple of list of names, and whether 
    or not there's doubts ('?') on the ancient name (0/1).
    In case there
    """
    names = []
    ancient_names = ""
    modern_names = ""
    if '?' in string:
        string = string.replace('?', '')
    if '(' in string:
        ancient_names = string.split('(')[0]
        modern_names = string.split('(')[1].split(')')[0]
        if '/' in modern_names:
            for name in modern_names.split('/'):
                names.append(name)
        else:
            names.append(modern_names)
    else:
        ancient_names = string
    
    if '/' in ancient_names:
        for name in ancient_names.split('/'):
            names.append(name)
    else:
        names.append(ancient_names)
    return names


# In[ ]:


location["Name (in list)"] = location["Name"].map(parse_names)
location.head()


# There are also columns with a question mark indicating if there's any doubt in the ancient and moden name equivalence, we will make a new column `Doubt?` indicating that information.

# In[ ]:


location["Doubt?"] = location["Name"].map(lambda string: '?' in string)
location.head()


# Now expand `location` into one name for each row.

# In[ ]:


location_expanded = location.explode("Name (in list)")
location_expanded.head()


# Now we have 5856 places.

# In[ ]:


location_expanded.shape


# Now we have both files (ANE file and our Drehem file) ready, we will match the geo names in these files. We will use the `fuzzywuzzy` library's `WRatio`, which is more tolerant to small differences like capitalization and punctuation. You can try different methods.
# 
# Note that the matching takes a fairly long time (> 30 mins) to run.

# In[ ]:


def fuzzy_search(geo_name):
  scores = location_expanded['Name (in list)'].map(lambda name: fuzz.WRatio(geo_name, name))
  
  return list(location_expanded[scores>90]['Name'])


# In[ ]:


non_emp_geo['fuzzy_match'] = non_emp_geo['cleaned_geo_name'].map(fuzzy_search)
non_emp_geo[non_emp_geo['fuzzy_match'].map(len)>0]


# If we set the matching score threshold to 90, we got 37 unique matches.

# In[ ]:


matches = non_emp_geo[non_emp_geo['fuzzy_match'].map(len)>0]
unique_matches = matches.drop_duplicates('geo_name').reset_index(drop=True)
# unique_matches = matched['cleaned_geo_name'].unique()
unique_matches.head()


# In[ ]:


unique_matches['fuzzy_match'] = unique_matches['fuzzy_match'].map(lambda l: l[0])
unique_matches.head()


# Here we saved the matches to drive so we can reuse it later.

# In[ ]:


non_emp_geo.to_csv("JupyterBook/jupyter_collection/04_Geographic_Names/working files/non_emp_geo.csv")


# Now we try to set the score to be > 87.5. Note the following cell also runs a fairly long time.

# In[ ]:


def fuzzy_search_875(geo_name):
  scores = location_expanded['Name (in list)'].map(lambda name: fuzz.WRatio(geo_name, name))
  
  return list(location_expanded[scores>87.5]['Name'])


# In[ ]:


non_emp_geo = pd.read_csv("JupyterBook/jupyter_collection/04_Geographic_Names/working files/non_emp_geo.csv")
non_emp_geo['fuzzy_match_875'] = non_emp_geo['cleaned_geo_name'].map(fuzzy_search_875)
non_emp_geo[non_emp_geo['fuzzy_match_875'].map(len)>0]


# We got more matches, but some of them are not what we want.

# In[ ]:


non_emp_geo[non_emp_geo['fuzzy_match_875'].map(len)>0]["cleaned_geo_name"].unique()


# We can see the subset of places which are not in the 33 matches with scores greater than 90 but are in matches with scores greater than 87.5. We can see some of them are not the same place, but there are places we missed in the 33.

# In[ ]:


in875_notin_90 = non_emp_geo[(non_emp_geo['fuzzy_match'].map(len)==2)&(non_emp_geo['fuzzy_match_875'].map(len)>0)]
in875_notin_90.head(50)


# In[ ]:


in875_notin_90[in875_notin_90["fuzzy_match_875"].map(lambda lst: any([s.find('Adamdun') != -1 for s in lst]))]


# Since manually going over the names takes a lot of time and labor, we will focus on the 33 good matches at this time.

# Here we will add the coordinate information to the matched locations.

# In[ ]:


def add_coordinates(str):
    s = location[location["Name"]==str]['geometry']
    if len(s) > 0:
        return s.iloc[0]
    else:
        return None

unique_matches['coordinates'] = unique_matches["fuzzy_match"].map(add_coordinates)
unique_matches = unique_matches.dropna().reset_index(drop=True)


# In[ ]:


unique_matches.head()


# Now we will use NetworkX to plot the locations in a graph. We have a few additional libraries to install and import.

# In[ ]:



get_ipython().system('pip install contextily')
get_ipython().system('pip install cartopy')
get_ipython().system('pip uninstall shapely')
get_ipython().system('pip install shapely --no-binary shapely')


# In[ ]:


from libpysal import weights, examples
from contextily import add_basemap
import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy


# GeoPandas dataframe requires a `geometry` attribute, we construct `geo_um`.

# In[ ]:


coordinates = np.column_stack((unique_matches["coordinates"].map(lambda p: p.x), unique_matches["coordinates"].map(lambda p: p.y)))


# In[ ]:


geo_um = gpd.GeoDataFrame(unique_matches, geometry = list(unique_matches['coordinates']))
geo_um.head()


# We try to build connections using p-numbers: since the same geographic name can appear in multiple p-numbers, so we add an edge if two places appear in the same p-number. But unfortunately, we can't find two places appearing in the same p-number, so instead we plotted a fully connected graph.

# In[ ]:


matches.head()


# In[ ]:


name_pnum = matches.groupby(['cleaned_geo_name'])['ip'].apply(list).reset_index(name = 'pnum')
name_pnum.head()


# In[ ]:


g = nx.Graph()

#Add nodes to the graph
for place in name_pnum['cleaned_geo_name']:
  g.add_node(place)

#Add edges
num_edges = 0
for place1 in name_pnum['cleaned_geo_name'].values:
  for place2 in name_pnum['cleaned_geo_name'].values:
    if place1 != place2:
      weight = 1
      # for pnum1 in name_pnum[name_pnum['cleaned_geo_name']==place1]['pnum'].values[0]:
      #   for pnum2 in name_pnum[name_pnum['cleaned_geo_name']==place2]['pnum'].values[0]:
      #     if pnum1 == pnum2:
      #       weight += 1
      #       num_edges += 1
      g.add_edge(place1, place2)

pos = {}
for place in name_pnum['cleaned_geo_name'].values:
  loc = unique_matches[unique_matches['cleaned_geo_name']==place]["coordinates"].values[0]
  pos[place] = (loc.x, loc.y)


# In[ ]:


crs = ccrs.PlateCarree()
fig, ax = plt.subplots(
    1, 1, figsize=(12, 8), subplot_kw=dict(projection=crs))
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.LAND, edgecolor='black')
ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
ax.add_feature(cartopy.feature.RIVERS)
ax.gridlines()
# Extent of .
ax.set_extent([30, 55, 25, 40])
nx.draw_networkx(g, ax=ax,
                 font_size=10,
                 alpha=.8,
                 width=.075,
                 pos=pos,
                 cmap=plt.cm.autumn)


# We can zoom in a little to the area where we have a lot of known geo names (around Drehem).

# In[ ]:


crs = ccrs.PlateCarree()
fig, ax = plt.subplots(
    1, 1, figsize=(12, 8), subplot_kw=dict(projection=crs))
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.LAND, edgecolor='black')
ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
ax.add_feature(cartopy.feature.RIVERS)
ax.gridlines()
# Extent of .
ax.set_extent([42, 48, 30, 35])
nx.draw_networkx(g, ax=ax,
                 font_size=10,
                 alpha=.8,
                 width=.075,
                 pos=pos,
                 cmap=plt.cm.autumn)


# This is in no way completed, potential next step might include interactive graphs like those realized in Gephi (potentially using Bokeh?) and explore other methods to set up edges (currently there are no two places with a same p number so I just simply connected them) and edge weights.
