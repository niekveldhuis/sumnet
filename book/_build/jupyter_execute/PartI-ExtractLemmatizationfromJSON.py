#!/usr/bin/env python
# coding: utf-8

# # I. Extract Lemmatization from JSON
# The code in this section will parse the [ORACC](http://oracc.org) JSON data of the Ur III corpus to extract lemmatization data. 
# 
# The output contains text IDs, line IDs, lemmas, and relevant other data. 

# In[1]:


#create necessary directories
import os
os.makedirs('jsonzip', exist_ok=True)
os.makedirs('output', exist_ok=True)

# import necessary libraries
import pandas as pd
from tqdm.auto import tqdm

# import libraries for this section
import requests
import zipfile
import json


# ## 1 Download ORACC Data
# 
# We will begin by downloading the lemmatization data from ORACC. This will be downloaded in the form of zipped JSON files which we will parse in the next subsection and store in a usable pandas dataframe. 
# 
# The ORACC database can be accessed via http://oracc.org/. As we are interested in the UR III data, we will use the project name to make the request. The project name corresponds to the last part of the url: http://oracc.org/epsd2/admin/ur3 and thus the project name is 'epsd2/admin/u3admin'. We are accessing the JSON through the API and so our request URL is http://build-oracc.museum.upenn.edu/json/epsd2-admin-ur3.
# 
# If you are unfamiliar with using APIs and would like to learn how to do so in Python, this section of the Data 100 textbook might be helpful: http://www.textbook.ds100.org/ch/13/web_http.html
# 
# **NOTE:** If you already have this data, you can skip this step as it can take quite a long time.

# In[2]:


project = 'epsd2/admin/ur3' #define project to download.

CHUNK = 1024 #define chunk size

proj = project.replace('/', '-')
url = "http://build-oracc.museum.upenn.edu/json/%s.zip" % proj
file_name = "jsonzip/%s.zip" % proj
with requests.get(url, stream=True) as request:
    if request.status_code == 200:
        #if file is accessible, download
        tqdm.write('Saving ' + url + ' as ' + file_name)
        total_size = int(request.headers.get('content-length', 0))
        tqdm_handler = tqdm(total=total_size, unit='B', unit_scale=True, desc = project)
        with open(file_name, 'wb') as zip_file:
        #use tdqm to show download speed
            for chunk in request.iter_content(chunk_size=CHUNK):
                tqdm_handler.update(len(chunk))
                zip_file.write(chunk)
    else:
        tqdm.write("WARNING: %s does not exist." % url)


# ## 2 Download BTDNS Data
# 
# Go to http://bdtns.filol.csic.es/index.php?p=formulario_urIII.
# 
# In the Provenience drop down menu choose "Puzriš-Dagan" and hit "Search". This will give a list of about 15,000 tablets from Drehem. 
# 
# Now, hit "Export" on the left and in the pop-up window select every box except "Transliteration of the whole text(s)". Hit "export". This will produce a .txt file and you'll need to put it into the "bdtns_metadata" folder.
# 
# The name of the .txt file changes everytime we download, so you'll need to change the following cell.

# ### 2.1 Extract Data from .txt file.

# In[3]:


bdtns_catalogue_data = pd.read_csv('https://raw.githubusercontent.com/niekveldhuis/sumnet/master/QUERY_catalogue.txt', delimiter='\t')
bdtns_catalogue_data = bdtns_catalogue_data.set_index('CDLI_no')
bdtns_catalogue_data


# ### 2.2 Filtering Data
# 
# Filter data for Puzriš-Dagān tablets and drop tablets with null p-numbers or without dates.

# In[4]:


bdtns_catalogue_data = bdtns_catalogue_data[(bdtns_catalogue_data['Provenience']
    =='Puzriš-Dagān') | (bdtns_catalogue_data['Provenience']
    =='Puzriš-Dagān (?)')] #select for Puzriš-Dagān

#select for non-null p_numbers and non-null dates
bdtns_catalogue_data = bdtns_catalogue_data.loc[~bdtns_catalogue_data.index.isnull(), :]
bdtns_catalogue_data = bdtns_catalogue_data[~bdtns_catalogue_data['Date'].isnull()]

bdtns_catalogue_data


# ## 3 Parsing the Data
# 
# In this subsection we will parse the imported `json` data into a list we can later use to construct a `pandas` dataframe.

# ### 3.1 The `parse_ORACC_json()` function
# Here we define the `parse_ORACC_json()` function, which will recursively parse the `json` files provided by ORACC until it finds the relevant lemma data. It takes in the an ORACC json data structure, the associated meta_data, and some necessary keys and returns a list of lemma python dictionary entries which we can use to constuct a pandas dataframe.
# 
# The `json` files consist of a hierarchy of `cdl` nodes; only the nodes which have 'f' or 'strict' keys contain lemmatization data. The function goes down this hierarchy by recursively calling itself when a new `cdl` node is encountered. For more information about the data hierarchy in the [ORACC](http://oracc.org) `json` files, see [ORACC Open Data](http://oracc.org/doc/opendata/index.html).
# 
# The argument of the `parse_ORACC_json()` function is a `JSON` object, essentially a Python dictionary that initially contains the entire contents of the original JSON file. The code works with a list of `JSON` dictionaries associated with the `cdl` key. Iterating through these dictionaries, if a dictionary contains another `cdl` node, the function calls itself with this lower-level dictionary as its argument `lemmatized_data_json`. Then the function will continue to recurse deeper into the data until it does not encounter any more `cdl` keys. Then, we are left with individual words. The code checks for a key `f`, and appendeds it's data as a `lemma` to the list `lemma_list`. The list `lemm_list`, which is initiated at the start of the function, is extended with the new lemmas at the end of each recursion, where each `lemma` dictionary represents a single word.
# 
# The variable `id_text` consists of a project abbreviation, such as `blms` or `cams/gkab` plus a text ID, in the format `cams/gkab/P338616` or `dcclt/Q000039`. 
# 
# The variable `ftype` is used to denote if a node is a year name (`yn`) or not. 
# 
# The field `word_id` consists of three parts, namely a text ID, line ID, and word ID, in the format `Q000039.76.2` meaning: the second word in line 76 of text object `Q000039`. Note that `76` is not a line number strictly speaking but an object reference within the text object. Things like horizontal rulings, columns, and breaks also get object references. The `word_id` field allows us to put lines, breaks, and horizontal drawings together in the proper order.
# 
# The field `label` is a human-legible label that refers to a line or another part of the text; it may look like `o i 23` (obverse column 1 line 23) or `r v 23'` (reverse column 5 line 23 prime). The `label` field is used in online [ORACC](http://oracc.org) editions to indicate line numbers. The `date`, `dates_references`, `primary_publication`, `collection`, and `museum_no` fields are also metadata which were taken from the ORACC catalogue.
# 
# The fields `extent`, `scope`, and `state` give metatextual data about the condition of the object; they capture the number of broken lines or columns and similar information. 

# In[5]:


def parse_ORACC_json(lemmatized_data_json, meta_data, dollar_keys):
    lemma_list = []
    for JSONobject in lemmatized_data_json['cdl']:
        if 'cdl' in JSONobject: 
            lemma_list.extend(parse_ORACC_json(JSONobject, meta_data, dollar_keys))
        if 'label' in JSONobject:
            meta_data['label'] = JSONobject['label']
        if 'f' in JSONobject:
            lemma = JSONobject['f']
            lemma['ftype'] = JSONobject.get('ftype')
            lemma['id_word'] = JSONobject['ref']
            lemma['label'] = meta_data['label']
            lemma['id_text'] = meta_data['id_text']
            lemma['date'] = meta_data['date']
            lemma['dates_references'] = meta_data['dates_references']
            lemma['publication'] = meta_data['publication']
            lemma['collection'] = meta_data['collection']
            lemma['museum_no'] = meta_data['museum_no']
            lemma['metadata_source'] = meta_data['metadata_source']
            lemma_list.append(lemma)
        if 'strict' in JSONobject and JSONobject['strict'] == '1':
            lemma = {key: JSONobject[key] for key in dollar_keys}
            lemma['id_word'] = JSONobject['ref']
            lemma['id_text'] = meta_data['id_text']
            lemma['date'] = meta_data['date']
            lemma['dates_references'] = meta_data['dates_references']
            lemma['publication'] = meta_data['publication']
            lemma['collection'] = meta_data['collection']
            lemma['museum_no'] = meta_data['museum_no']
            lemma['metadata_source'] = meta_data['metadata_source']
            lemma_list.append(lemma)
    return lemma_list


# ### 3.2 Getting Catalogue Data for the  ORACC Drehem Tablets.
# 
# Before constructing our dataframe of lemmatized data, we will construct a dataframe of Drehem tablet metadata to filter and augment the data with. 
# 
# We do this by loading the data from the `catalogue.json` into a dataframe and selecting the pnumbers where the provenience is indicated to be Puzriš-Dagan or modern Drehem.

# In[6]:


zip_file = zipfile.ZipFile("jsonzip/%s.zip" % proj)    # create a Zipfile object

files = zip_file.namelist()
oracc_json_str = zip_file.read("%s/catalogue.json" % project).decode('utf-8') #read and decode the json file of one particular text
oracc_catalogue_json = json.loads(oracc_json_str)['members']

zip_file.close()

#create combined catalogue_data
oracc_catalogue_data = pd.DataFrame.from_dict(oracc_catalogue_json, orient='index')
catalogue_data = oracc_catalogue_data[(oracc_catalogue_data['provenience'] 
    == 'Puzriš-Dagan (mod. Drehem)') | 
    oracc_catalogue_data.index.isin(bdtns_catalogue_data.index)]
    #choose catalogue entries marked as Puzriš-Dagan in either ORACC or BDTNS

#add bdtns dates to oracc dataset and name metadata source
#added line to suppress setting with copy warning
pd.options.mode.chained_assignment = None  # default='warn'
in_bdtns_index = catalogue_data.index.isin(bdtns_catalogue_data.index)
catalogue_data.loc[in_bdtns_index, 'date_of_origin'] = bdtns_catalogue_data['Date']
catalogue_data.loc[in_bdtns_index, 'dates_referenced'] = bdtns_catalogue_data['Date']
catalogue_data.loc[in_bdtns_index, 'provenience'] = bdtns_catalogue_data['Provenience']
catalogue_data['metadata_source'] = ['BDTNS' if in_bdtns else 'ORACC' for in_bdtns in in_bdtns_index]


catalogue_data = catalogue_data[['date_of_origin', 'dates_referenced', 'collection',
                                 'primary_publication', 'museum_no', 'provenience',
                                 'metadata_source']]

catalogue_data


# ### 3.3 Call the `parse_ORACC_json()` function for every `JSON` file
# Here we will use the `parse_ORACC_json()` and the `catalogue` dataframe to construct the `words_df` dataframe which we will be using throughout the rest of the code. The project `zip` file downloaded earlier contains a directory that is called `corpusjson` that contains a JSON file for every text that is available in that corpus. The files are named using their text IDs in the pattern `P######.json` (or `Q######.json` or `X######.json`).
# 
# The function `namelist()` of the `zipfile` package is used to create a list of the names of all the files in the ZIP. From this list we select all the file names in the `corpusjson` directory with extension `.json` (this way we exclude the name of the directory itself). 
# 
# Each of these files is read from the `zip` file and loaded with the command `json.loads()`, which transforms the string into a proper JSON object. 
# 
# This JSON object (essentially a Python dictionary), which is called `data_json` is now sent to the `parse_ORACC_json()` function. The function adds lemmata to the `lemm_l` list. In the end, `lemm_l` will contain as many list elements as there are words in all the texts in the projects requested.
# 
# The dictionary `meta_d` is created to hold temporary information. The value of the key `id_text` is updated in the main process every time a new JSON file is opened and send to the `parsejson()` function. The `parsejson()` function itself will change values or add new keys, depending on the information found while iterating through the JSON file. When a new lemma row is created, `parse_ORACC_json()` will supply data such as `id_text`, `label` and (potentially) other information from `meta_d`.

# In[7]:


lemma_list = []
meta_data = {'label': None}
dollar_keys = ['extent', 'scope', 'state']

try:
    zip_file = zipfile.ZipFile(file_name)       # create a Zipfile object
except:
    print(file_name, 'does not exist or is not a proper ZIP file')

drehem_list_oracc = catalogue_data.index.to_list()
files = [project + '/corpusjson/' + p_number + '.json' for p_number in drehem_list_oracc]    #that holds all the P, Q, and X numbers.
for lemma_file in tqdm(files, desc = project):       #iterate over the file names
    id_text = project + lemma_file[-13:-5] # id_text is, for instance, blms/P414332

    p_number = lemma_file[-12:-5]
    meta_data['id_text'] = id_text
    meta_data['date'] = catalogue_data.loc[p_number]['date_of_origin']
    meta_data['dates_references'] = catalogue_data.loc[p_number]['dates_referenced']
    meta_data['publication'] = catalogue_data.loc[p_number]['primary_publication']
    meta_data['collection'] = catalogue_data.loc[p_number]['collection']
    meta_data['museum_no'] = catalogue_data.loc[p_number]['museum_no']
    meta_data['metadata_source'] = catalogue_data.loc[p_number]['metadata_source']

    try:
        json_str = zip_file.read(lemma_file).decode('utf-8')         #read and decode the json file of one particular text
        lemmatized_data_json = json.loads(json_str)                # make it into a json object (essentially a dictionary)
        lemma_list.extend(parse_ORACC_json(lemmatized_data_json, meta_data, dollar_keys))               # and send to the parse_ORACC_json() function
    except:
        print(id_text, 'is not available or not complete')

zip_file.close()


# ## 4 Data Structuring
# 
# Here we construct our `pandas` dataframe.
# 

# ### 4.1 Transform the Data into a DataFrame
# Here we use `lemma_list` to make our dataframe and we view it.

# In[8]:


words_df = pd.DataFrame(lemma_list)
words_df = words_df.fillna('')      # fill empty values with empty strings
words_df[words_df['lang'] == 'sux'] # display lemmas where language is Sumerian


# In[9]:


words_df.shape   # see shape


# In[10]:


words_df.columns # see column names


# ### 4.2 Remove Spaces and Commas from Guide Word and Sense
# Spaces and commas in Guide Word and Sense may cause trouble in computational methods in tokenization, or when saved in Comma Separated Values format. All spaces and commas are replaced by hyphens and nothing (empty string), respectively.
# 
# By default the `replace()` function in `pandas` will match the entire string (that is, "lugal" matches "lugal" but there is no match between "l" and "lugal"). In order to match partial strings the parameter `regex` must be set to `True`.
# 
# The `replace()` function takes a nested dictionary as argument. The top-level keys identify the columns on which the `replace()` function should operate (in this case 'gw' and 'sense'). The value of each key is another dictionary with the search string as key and the replace string as value.

# In[11]:


findreplace = {' ' : '-', ',' : ''}
words_df = words_df.replace({'gw' : findreplace, 'sense' : findreplace}, regex=True)


# The columns in the resulting DataFrame correspond to the elements of a full [ORACC](http://oracc.org) signature, plus information about text, line, and word ids:
# * base (Sumerian only)
# * cf (Citation Form)
# * cont (continuation of the base; Sumerian only)
# * epos (Effective Part of Speech)
# * form (transliteration, omitting all flags such as indication of breakage)
# * frag (transliteration; including flags)
# * gdl_utf8 (cuneiform)
# * gw (Guide Word: main or first translation in standard dictionary)
# * id_text (six-digit P, Q, or X number)
# * id_word (word ID in the format Text_ID.Line_ID.Word_ID)
# * label (traditional line number in the form o ii 2' (obverse column 2 line 2'), etc.)
# * lang (language code, including sux, sux-x-emegir, sux-x-emesal, akk, akk-x-stdbab, etc)
# * morph (Morphology; Sumerian only)
# * norm (Normalization: Akkadian)
# * norm0 (Normalization: Sumerian)
# * pos (Part of Speech)
# * sense (contextual meaning)
# * sig (full ORACC signature)
# * (newly added) ftype (`yn` if a node represents a year name)
# 
# Not all data elements (columns) are available for all words. Sumerian words never have a `norm`, Akkadian words do not have `norm0`, `base`, `cont`, or `morph`. Most data elements are only present when the word is lemmatized; only `lang`, `form`, `id_word`, and `id_text` should always be there.
# 
# We also included some metadata:
# 
# * date
# * dates_references
# * primary_publication
# * collection
# * museum_no
# 
# The first two of which refer to the dates on the tablets and the last three of which refer to the current ownership and where the tablet was published.

# ### 4.3 Create Line ID
# The DataFrame currently has a word-by-word data representation. We will add to each word a field `id_line` that will make it possible to reconstruct lines. This newly created field `id_line` is different from a traditional line number (found in the field "label") in two ways. First, id_line is an integer, so that lines are sorted correctly. Second, `id_line` is assigned to words, but also to gaps and horizontal drawings on the tablet. The field `id_line` will allow us to keep all these elements in their proper order.
# 
# The field "id_line" is created by splitting the field "id_word" into (two or) three elements. The format of "id_word" is IDtext.line.word. The middle part, id_line, is selected and its data type is changed from string to integer. Rows that represent gaps in the text or horizontal drawings have an "id_word" in the format IDtext.line (consisting of only two elements), but are treated in exactly the same way.

# In[12]:


words_df['id_line'] = [int(wordid.split('.')[1]) for wordid in words_df['id_word']]
words_df.head(10)


# ## 5 Create Lemma Column
# A lemma, [ORACC](http://oracc.org) style, combines Citation Form, GuideWord and POS into a unique reference to one particular lemma in a standard dictionary, as in `lugal[king]N` (Sumerian) or `šarru[king]N`. 
# 
# Proper Nouns (proper names, geographical names, etc.) are a special case, because they currently receive a Part of Speech, but not a Citation Form or Guide Word.
# 
# Usually, not all words in a text are lemmatized, because a word may be (partly) broken and/or unknown. Unlemmatized and unlemmatizable words will receive a place-holder lemmatization that consists of the transliteration of the word (instead of the Citation Form), with `NA` as GuideWord and `NA` as POS, as in `i-bu-x[NA]NA`. Note that `NA` is a string. 
# 
# Finally, rows representing horizontal rulings, blank lines, or broken lines have data in the fields 'state', 'scope', and 'extent' (for instance 'state' = broken, 'scope' = line, and 'extent' = 3, to indicate three broken lines). We can use this to prevent scripts to 'jump over' such breaks when looking for key words after (or before) a proper noun. We distinguish between physical breaks and logical breaks (such as horizontal rulings or seal impressions). 

# In[13]:


proper_nouns = ['FN', 'PN', 'DN', 'AN', 'WN', 'ON', 'TN', 'MN', 'CN', 'GN']
physical_break = ['illegible', 'traces', 'missing', 'effaced']
logical_break = ['other', 'blank', 'ruling']
words_df['lemma'] = words_df["cf"] + '[' + words_df["gw"] + ']' + words_df["pos"]
words_df.loc[words_df["cf"] == "" , 'lemma'] = words_df['form'] + '[NA]NA'
words_df.loc[words_df["pos"] == "n" , 'lemma'] = words_df['form'] + '[]NU' 
#words_df.loc[words_df["pos"].isin(proper_nouns) , 'lemma'] = words_df['form'] + '[]' + words_df['pos']
words_df.loc[words_df["state"].isin(logical_break), 'lemma'] = "break_logical"
words_df.loc[words_df["state"].isin(physical_break), 'lemma'] = "break_physical"
words_df.head(10)


# ## 6 Select Relevant Columns
# Now we can select only the field 'lemma' and the fields that indicate the ID of the word, the line, or the document, plus the metadata, and the field 'label', which indicates the physical position of the line on the document.

# In[14]:


cols = ['lemma', 'id_text', 'id_line', 'id_word', 'label', 'date', 'dates_references', 'publication', 'collection', 'museum_no', 'ftype', 'metadata_source']
words_df = words_df[cols].copy()
words_df.head(100)


# We can simplify the 'id_text' column, because all documents derive from the same project (epsd2/admnin/u3adm).

# In[15]:


words_df['id_text'] = [tid[-7:] for tid in words_df['id_text']]
words_df.head(20)


# In[16]:


test = words_df.head(4000)
test[test['lemma'].str.contains('N')]


# ## 7 Save Results in CSV file & Pickle
# The output file is called `part_1_output.csv` and is placed in the directory `output`. In most computers, `csv` files open automatically in Excel. This program does not deal well with `utf-8` encoding (files in `utf-8` need to be imported; see the instructions [here](https://www.itg.ias.edu/content/how-import-csv-file-uses-utf-8-character-encoding-0). If you intend to use the file in Excel, change `encoding ='utf-8'` to `encoding='utf-16'`. For usage in computational text analysis applications `utf-8` is usually preferred. 
# 
# The Pandas function `to_pickle()` writes a binary file that can be opened in a later phase of the project with the `read_pickle()` command and will reproduce exactly the same DataFrame with the same data structure. The resulting file can only be used by pandas.
# 
# You can skip Part I by importing the Part I output as a pandas dataframe at the beginning of Part II. Similarly, in the next parts, after you save the output, you will be able to import that output later on instead of rerunning the sections before.

# In[17]:


words_df.to_csv('output/part_1_output.csv')
words_df.to_pickle('output/part_1_output.p')

catalogue_data.to_csv('output/part_1_catalogue.csv')
catalogue_data.to_pickle('output/part_1_catalogue.p')

