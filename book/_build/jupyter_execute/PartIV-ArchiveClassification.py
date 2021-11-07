#!/usr/bin/env python
# coding: utf-8

# # IV. Archive Classification
# 
# The primary question which we pursue in this section is how one can use reproducible and replicable workflows for discovering the optimal classifications of the text groups from the Drehem texts, found in an unprovenanced archival context. We describe how we leverage existing classification models to help validate our findings. 

# In[ ]:


# import necessary libraries
import pandas as pd
from tqdm.auto import tqdm

# import libraries for this section
import re
import matplotlib.pyplot as plt

# import ML models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
# import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ## 1 Set up Data
# Create a dictionary of archive categories

# ### 1.1 Labeling the Training Data
# 
# We will be labeling the data according to what words show up in it.

# In[ ]:


labels = dict()
labels['domesticated_animal'] = ['ox', 'cow', 'sheep', 'goat', 'lamb', '~sheep', 'equid'] # account for plural
#split domesticated into large and small - sheep, goat, lamb, ~sheep would be small domesticated animals
labels['wild_animal'] = ['bear', 'gazelle', 'mountain'] # account for 'mountain animal' and plural
labels['dead_animal'] = ['die'] # find 'die' before finding domesticated or wild
labels['leather_object'] = ['boots', 'sandals']
labels['precious_object'] = ['copper', 'bronze', 'silver', 'gold']
labels['wool'] = ['wool', '~wool']
# labels['queens_archive'] = []


# Using filtered_with_neighbors.csv generated above, make P Number and id_line the new indices.
# 
# Separate components of the lemma.

# In[ ]:


words_df = pd.read_pickle('https://gitlab.com/yashila.bordag/sumnet-data/-/raw/main/part_3_words_output.p') # uncomment to read from online file
#words_df = pd.read_pickle('output/part_3_output.p') #uncomment to read from local file


# In[ ]:


data = words_df.copy()
data.loc[:, 'pn'] = data.loc[:, 'id_text'].str[-6:].astype(int)
data = data.set_index(['pn', 'id_line']).sort_index()
extracted = data.loc[:, 'lemma'].str.extract(r'(\S+)\[(.*)\](\S+)')
data = pd.concat([data, extracted], axis=1)
data = data.fillna('') #.dropna() ????
data.head()


# In[ ]:


data['label'].value_counts()


# In[ ]:


for archive in labels.keys():
    data.loc[data.loc[:, 1].str.contains('|'.join([re.escape(x) for x in labels[archive]])), 'archive'] = archive

data.loc[:, 'archive'] = data.loc[:, 'archive'].fillna('')

data.head()


# The function get_set has a dataframe row as an input and returns a dictionary where each key is a word type like NU and PN. The values are its corresponding lemmas.

# ### 1.2 Data Structuring

# In[ ]:


def get_set(df):
    
    d = {}

    seals = df[df['label'].str.contains('seal')]
    df = df[~df['label'].str.contains('seal')]

    for x in df[2].unique():
        d[x] = set(df.loc[df[2] == x, 0])

    d['SEALS'] = {}
    for x in seals[2].unique():
        d['SEALS'][x] = set(seals.loc[seals[2] == x, 0])

    return d


# In[ ]:


get_set(data.loc[100271])


# In[ ]:


archives = pd.DataFrame(data.groupby('pn').apply(lambda x: set(x['archive'].unique()) - set(['']))).rename(columns={0: 'archive'})
archives.loc[:, 'set'] = data.reset_index().groupby('pn').apply(get_set)
archives.loc[:, 'archive'] = archives.loc[:, 'archive'].apply(lambda x: {'dead_animal'} if 'dead_animal' in x else x)
archives.head()


# In[ ]:


def get_line(row, pos_lst=['N']):
    words = {'pn' : [row.name]} #set p_number
    for pos in pos_lst:
        if pos in row['set']:
            #add word entries for all words of the selected part of speech
            words.update({word: [1] for word in row['set'][pos]})
    return pd.DataFrame(words)


# Each row represents a unique P-number, so the matrix indicates which word are present in each text.

# In[ ]:


sparse = words_df.groupby(by=['id_text', 'lemma']).count()
sparse = sparse['id_word'].unstack('lemma')
sparse = sparse.fillna(0)


# In[ ]:


sparse = pd.concat(archives.apply(get_line, axis=1).values).set_index('pn')

sparse


# In[ ]:


sparse = sparse.fillna(0)
sparse = sparse.join(archives.loc[:, 'archive'])


# In[ ]:


sparse.loc[sparse.loc[:, 'archive'].apply(lambda x: 'domesticated_animal' in x), 'domesticated_animal'] = 1
sparse.loc[:, 'domesticated_animal'] = sparse.loc[:, 'domesticated_animal'].fillna(0)

sparse.loc[sparse.loc[:, 'archive'].apply(lambda x: 'wild_animal' in x), 'wild_animal'] = 1
sparse.loc[:, 'wild_animal'] = sparse.loc[:, 'wild_animal'].fillna(0)

sparse.loc[sparse.loc[:, 'archive'].apply(lambda x: 'dead_animal' in x), 'dead_animal'] = 1
sparse.loc[:, 'dead_animal'] = sparse.loc[:, 'dead_animal'].fillna(0)

sparse.loc[sparse.loc[:, 'archive'].apply(lambda x: 'leather_object' in x), 'leather_object'] = 1
sparse.loc[:, 'leather_object'] = sparse.loc[:, 'leather_object'].fillna(0)

sparse.loc[sparse.loc[:, 'archive'].apply(lambda x: 'precious_object' in x), 'precious_object'] = 1
sparse.loc[:, 'precious_object'] = sparse.loc[:, 'precious_object'].fillna(0)

sparse.loc[sparse.loc[:, 'archive'].apply(lambda x: 'wool' in x), 'wool'] = 1
sparse.loc[:, 'wool'] = sparse.loc[:, 'wool'].fillna(0)
sparse.head()


# In[ ]:


known = sparse.loc[sparse['archive'].apply(len) == 1, :]
unknown = sparse.loc[(sparse['archive'].apply(len) == 0) | (sparse['archive'].apply(len) > 1), :]


# In[ ]:


unknown_0 = sparse.loc[(sparse['archive'].apply(len) == 0), :]


# In[ ]:


unknown.shape


# ### 1.3 Data Exploration

# In[ ]:


unknown.loc[sparse['archive'].apply(len) > 1, :]


# In[ ]:


#find rows where archive has empty set
unknown[unknown['archive'] == set()]


# In[ ]:


known_copy = known
known_archives = [known_copy['archive'].to_list()[i].pop() for i in range(len(known_copy['archive'].to_list()))]
known_archives


# In[ ]:


known['archive_class'] = known_archives


# In[ ]:


archive_counts = known['archive_class'].value_counts()


plt.xlabel('Archive Class')
plt.ylabel('Frequency', rotation=0, labelpad=30)
plt.title('Frequencies of Archive Classes in Known Archives')
plt.xticks(rotation=45)
plt.bar(archive_counts.index, archive_counts);

percent_domesticated_animal = archive_counts['domesticated_animal'] / sum(archive_counts)

print('Percent of texts in Domesticated Animal Archive:', percent_domesticated_animal)


# In[ ]:


known.shape


# In[ ]:


words_df_copy = words_df.copy()
words_df_copy['id_text'] = [int(pn[1:]) for pn in words_df_copy['id_text']]

grouped = words_df_copy.groupby(by=['id_text']).first()
grouped = grouped.fillna(0)

known_copy = known.copy()
known_copy['year'] = grouped.loc[grouped.index.isin(known.index),:]['min_year']

year_counts = known_copy.groupby(by=['year', 'archive_class'], as_index=False).count().set_index('year').loc[:, 'archive_class':'ki']
year_counts_pivoted = year_counts.pivot(columns='archive_class', values='ki').fillna(0)


# In[ ]:


year_counts_pivoted.drop(index=0).plot();
plt.xlabel('Year')
plt.ylabel('Count', rotation=0, labelpad=30)
plt.title('Frequencies of Archive Classes in Known Archives over Time');


# In[ ]:


known


# In[ ]:


#run to save the prepared data
known.to_csv('output/part_4_known.csv')
known.to_pickle('output/part_4_known.p')
unknown.to_csv('output/part_4_unknown.csv')
unknown.to_pickle('output/part_4_unknown.p')
unknown_0.to_csv('output/part_4_unknown_0.csv')
unknown_0.to_pickle('output/part_4_unknown_0.p')


# In[ ]:


#known = pd.read_pickle('https://gitlab.com/yashila.bordag/sumnet-data/-/raw/main/part_4_known.p')
#unknown = pd.read_pickle('https://gitlab.com/yashila.bordag/sumnet-data/-/raw/main/part_4_unknown.p')
#unknown_0 = pd.read_pickle('https://gitlab.com/yashila.bordag/sumnet-data/-/raw/main/part_4_unknown_0.p')

model_weights = {}


# #### 1.3.1 PCA/Dimensionality Reduction
# 
# Here we perform PCA to find out more about the underlying structure of the dataset. We will analyze the 2 most important principle components and explore how much of the variation of the known set is due to these components.

# In[ ]:


#PCA
pca_archive = PCA()
principalComponents_archive = pca_archive.fit_transform(known.loc[:, 'AN.bu.um':'šuʾura'])


# In[ ]:


principal_archive_Df = pd.DataFrame(data = principalComponents_archive
             , columns = ['principal component ' + str(i) for i in range(1, 1 + len(principalComponents_archive[0]))])


# In[ ]:


len(known.loc[:, 'AN.bu.um':'šuʾura'].columns)


# In[ ]:


principal_archive_Df


# In[ ]:


principal_archive_Df.shape


# In[ ]:


print('Explained variation per principal component: {}'.format(pca_archive.explained_variance_ratio_))


# In[ ]:


plt.plot(pca_archive.explained_variance_ratio_)


# In[ ]:


known_reindexed = known.reset_index()
known_reindexed


# In[ ]:


plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component 1',fontsize=20)
plt.ylabel('Principal Component 2',fontsize=20)
plt.title("Principal Component Analysis of Archives",fontsize=20)
targets = ['domesticated_animal', 'wild_animal', 'dead_animal', 'leather_object', 'precious_object', 'wool']
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
for target, color in zip(targets,colors):
    indicesToKeep = known_reindexed.index[known_reindexed['archive_class'] == target].tolist()
    plt.scatter(principal_archive_Df.loc[indicesToKeep, 'principal component 1']
               , principal_archive_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})

# import seaborn as sns
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="principal component 1", y="principal component 2",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=principal_cifar_Df,
#     legend="full",
#     alpha=0.3
# )


# ## 2 Simple Modeling Methods

# ### 2.1 Logistic Regression
# 
# Here we will train our model using logistic regression to predict archives based on the features made in the previous subsection.

# #### 2.1.1 Logistic Regression by Archive
# 
# Here we will train and test a set of 1 vs all Logistic Regression Classifiers which will attempt to classify tablets as either a part of an archive, or not in an archive.

# In[ ]:


clf_da = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
clf_da.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'domesticated_animal'])
clf_da.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'domesticated_animal'])


# In[ ]:


clf_wa = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
clf_wa.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'wild_animal'])
clf_wa.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'wild_animal'])


# In[ ]:


clf_dea = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
clf_dea.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'dead_animal'])
clf_dea.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'dead_animal'])


# In[ ]:


clf_lo = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
clf_lo.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'leather_object'])
clf_lo.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'leather_object'])


# In[ ]:


clf_po = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
clf_po.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'precious_object'])
clf_po.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'precious_object'])


# In[ ]:


clf_w = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
clf_w.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'wool'])
clf_w.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'wool'])


# In[ ]:


known.loc[:, 'AN.bu.um':'šuʾura']


# As we can see the domesticated animal model has the lowest accuracy while the leather object, precious_object, and wool classifiers work fairly well.

# #### 2.1.2 Multinomial Logistic Regression
# 
# Here we will be using multinomial logistic regression as we have multiple archive which we could classify each text into. We are fitting our data onto the tablets with known archives and then checking the score to see how accurate the model is.
# 
# Finally, we append the Logistic Regression prediction as an archive prediction for the tablets without known archives.

# In[ ]:


clf_archive = LogisticRegression(random_state=42, solver='lbfgs', max_iter=300)
clf_archive.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'archive_class'])
log_reg_score = clf_archive.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'archive_class'])
model_weights['LogReg'] = log_reg_score
log_reg_score


# In[ ]:


#Predictions for Unknown
unknown["LogReg Predicted Archive"] = clf_archive.predict(unknown.loc[:, 'AN.bu.um':'šuʾura'])
unknown


# In[ ]:


known['archive_class'].unique()


# ### 2.2 K Nearest Neighbors
# 
# Here we will train our model using k nearest neighbors to predict archives based on the features made in the previous subsection. We are fitting our data onto the tablets with known archives and then checking the score to see how accurate the model is.
# 
# We then append the KNN prediction as an archive prediction for the tablets without known archives.
# 
# Then, we use different values for K (the number of neighbors we take into consideration when predicting for a tablet) to see how the accuracy changes for different values of K. This can be seen as a form of hyperparameter tuning because we are trying to see which K we should choose to get the highest training accuracy.

# In[ ]:


#takes long time to run, so don't run again
list_k = [3, 5, 7, 9, 11, 13]
max_k, max_score = 0, 0
for k in list_k:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'archive_class'])
    knn_score = knn.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'archive_class'])
    print("Accuracy for k = %s: " %(k), knn_score)
    if max_score <= knn_score:
        max_score = knn_score
        max_k = k
    


# As we can see here, k = 5 and k = 9 have the best training accuracy performance which falls roughly in line with the Logistic Regression classification training accuracy.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=max_k)
knn.fit(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'archive_class'])
knn_score = knn.score(known.loc[:, 'AN.bu.um':'šuʾura'], known.loc[:, 'archive_class'])
model_weights['KNN'] = knn_score


# In[ ]:


#Predictions for Unknown
unknown["KNN Predicted Archive"] = knn.predict(unknown.loc[:, 'AN.bu.um':'šuʾura'])
unknown


# As we can see in the output from the previous cell, we can get different predictions depending on the classifier we choose.

# Next we will split the data we have on tablets with known archives into a test and training set to further understant the atraining accuracy. For the next two sections, we will use `X_train` and `y_train` to train the data and `X_test` and `y_test` to test the data. As the known set was split randomly, we presume that both the training and test set are representative of the whole known set, so the two sets are reasonably comparable.

# In[ ]:


#Split known into train and test, eventually predict with unknown 
X_train, X_test, y_train, y_test = train_test_split(known.loc[:, 'AN.bu.um':'šuʾura'], 
                                                    known.loc[:, 'archive_class'], 
                                                    test_size=0.2,random_state=0) 


# ### 2.3 Naive Bayes
# 
# Here we will train our model using a Naive Bayes Model to predict archives based on the features made in the previous subsection. Here, we make the assumption that the features are independent of each other, from which the descriptor _naive_ comes from. So:
# 
# $$P(x_i|y; x_1, x_2, ..., x_{i-1}, x_{i+1}, ..., x_n) = P(x_i| y)$$
# 
# and:
# 
# $$P(x_1, x_2, ..., x_n | y) = \prod_{i=1}^{n} P(x_i | y)$$
# 
# Moreover, we will be using Bayes' Law, which in this case states:
# 
# $$P(y|x_1, x_2, ..., x_n) = \frac{P(y)P(x_1, x_2, ..., x_n | y)}{P(x_1, x_2, ..., x_n)}$$
# 
# eg. the probability of a particular tablet (defined by features $x_1, x_2, ..., x_n$) is in archive $y$, is equal to the probability of getting a tablet from archive $y$ times the probability you would get a particular set of features $x_1, x_2, ..., x_n$ divided by the probability of getting a particular set of features $x_1, x_2, ..., x_n$.
# 
# Applying our assumption of independence from before, we can simplify this to:
# 
# $$P(y|x_1, x_2, ..., x_n) = \frac{P(y)\prod_{i=1}^{n} P(x_i | y)}{P(x_1, x_2, ..., x_n)}$$
# 
# Which means the probability of a particular tablet (defined by features $x_1, x_2, ..., x_n$) is in archive $y$ is _proportional_ to 
# 
# $$P(y|x_1, x_2, ..., x_n) \propto P(y)\prod_{i=1}^{n} P(x_i | y)$$ probability of getting a tablet from archive $y$ times the product of probabilities of getting a feature $x_i$ given an archive $y$.
# 
# We can then use this to calculate the maximizing archive.
# 
# $$\hat{y} = \underset{y}{argmax} \; P(y)\prod_{i=1}^{n} P(x_i | y)$$
# 
# We are training two models where the first assumes the features are Gaussian random variables and the second assumes the features are Bernoulli random variables.
# 
# We are fitting our data onto the tablets with known archives and then checking the score to see how accurate the model is.
# 
# Finally, we append the two Naive Bayes predictions as archive predictions for the tablets without known archives.

# In[ ]:


#Gaussian
gauss = GaussianNB()
gauss.fit(X_train, y_train)
gauss_nb_score = gauss.score(X_test, y_test)
model_weights['GaussNB'] = gauss_nb_score
gauss_nb_score


# We can see than the Gaussian assumption does quite poorly.

# In[ ]:


#Predictions for Unknown
unknown["GaussNB Predicted Archive"] = gauss.predict(unknown.loc[:, 'AN.bu.um':'šuʾura'])
unknown


# In[ ]:


#Bernoulli
bern = BernoulliNB()
bern.fit(X_train, y_train)
bern_nb_score = bern.score(X_test, y_test)
model_weights['BernoulliNB'] = bern_nb_score
bern_nb_score


# However the Bernoulli assumption does quite well.

# In[ ]:


#Predictions for Unknown
unknown["BernoulliNB Predicted Archive"] = bern.predict(unknown.loc[:, 'AN.bu.um':'šuʾura'])
unknown


# ### 2.4 SVM
# 
# Here we will train our model using Support Vector Machines to predict archives based on the features made earlier in this section. We are fitting our data onto the tablets with known archives and then checking the score to see how accurate the model is.
# 
# Finally, we append the SVM prediction as an archive prediction for the tablets without known archives.

# In[ ]:


svm_archive = svm.SVC(kernel='linear')
svm_archive.fit(X_train, y_train)
y_pred = svm_archive.predict(X_test)
svm_score = metrics.accuracy_score(y_test, y_pred)
model_weights['SVM'] = svm_score
print("Accuracy:", svm_score)


# In[ ]:


unknown["SVM Predicted Archive"] = svm_archive.predict(unknown.loc[:, 'AN.bu.um':'šuʾura'])
unknown


# ## 3 Complex Modeling Methods

# ## 4 Voting Mechanism Between Models
# 
# Here we will use the models to determine the archive which to assign to each tablet with an unknown archive. 
# 
# We will then augment the words_df with these archives.

# In[ ]:


model_weights


# In[ ]:


def visualize_archives(data, prediction_name):
    archive_counts = data.value_counts()


    plt.xlabel('Archive Class')
    plt.ylabel('Frequency', rotation=0, labelpad=30)
    plt.title('Frequencies of ' + prediction_name + ' Predicted Archives')
    plt.xticks(rotation=45)
    plt.bar(archive_counts.index, archive_counts);

    percent_domesticated_animal = archive_counts['domesticated_animal'] / sum(archive_counts)

    print('Percent of texts in Domesticated Animal Archive:', percent_domesticated_animal)


# In[ ]:


#Log Reg Predictions
visualize_archives(unknown['LogReg Predicted Archive'], 'Logistic Regression')


# In[ ]:


#KNN Predictions
visualize_archives(unknown['KNN Predicted Archive'], 'K Nearest Neighbors')


# In[ ]:


#Gaussian Naive Bayes Predictions
visualize_archives(unknown['GaussNB Predicted Archive'], 'Gaussian Naive Bayes')


# In[ ]:


#Bernoulli Naive Bayes Predictions
visualize_archives(unknown['BernoulliNB Predicted Archive'], 'Bernoulli Naive Bayes')


# In[ ]:


#SVM Predictions
visualize_archives(unknown['SVM Predicted Archive'], 'Support Vector Machines Naive Bayes')


# In[ ]:


def weighted_voting(row):
    votes = {} # create empty voting dictionary
    # tally votes
    for model in row.index:
        model_name = model[:-18] # remove ' Predicted Archive' from column name
        prediction = row[model]
        if prediction not in votes.keys():
            votes[prediction] = model_weights[model_name] # if the prediction isn't in the list of voting categories, add it with a weight equal to the current model weight 
        else:
            votes[prediction] += model_weights[model_name] # else, add model weight to the prediction
    return max(votes, key=votes.get) # use the values to get the prediction with the greatest weight


# In[ ]:


predicted_archives = unknown.loc[:, 'LogReg Predicted Archive':
                                   'SVM Predicted Archive'].copy() # get predictions
weighted_prediction = predicted_archives.apply(weighted_voting, axis=1) #apply voting mechanism on each row and return 'winning' prediction


# In[ ]:


weighted_prediction[weighted_prediction != 'domesticated_animal']


# In[ ]:


words_df


# In[ ]:


archive_class = known['archive_class'].copy().append(weighted_prediction)
words_df['archive_class'] = words_df.apply(lambda row: archive_class[int(row['id_text'][1:])], axis=1)


# In[ ]:


words_df


# ## 5 Sophisticated Naive Bayes

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ### 5.1 Feature and Model Creation

# There are some nouns that are so closely associated with a specific archive that their presence in a text virtually guarantees that the text belongs to that archive. We will use this fact to create a training set for our classification model.
# 
# The `labels` dictionary below contains the different archives along with their possible associated nouns.

# In[ ]:


labels = dict()
labels['domesticated_animal'] = ['ox', 'cow', 'sheep', 'goat', 'lamb', '~sheep', 'equid']
dom = '(' + '|'.join(labels['domesticated_animal']) + ')'
#split domesticated into large and small - sheep, goat, lamb, ~sheep would be small domesticated animals
labels['wild_animal'] = ['bear', 'gazelle', 'mountain', 'lion'] # account for 'mountain animal' and plural
wild = '(' + '|'.join(labels['wild_animal']) + ')'
labels['dead_animal'] = ['die'] # find 'die' before finding domesticated or wild
dead = '(' + '|'.join(labels['dead_animal']) + ')'
labels['leather_object'] = ['boots', 'sandals']
leath = '(' + '|'.join(labels['leather_object']) + ')'
labels['precious_object'] = ['copper', 'bronze', 'silver', 'gold']
prec = '(' + '|'.join(labels['precious_object']) + ')'
labels['wool'] = ['wool', '~wool', 'hair']
wool = '(' + '|'.join(labels['wool']) + ')'
complete = []
for lemma_list in labels.values():
  complete = complete + lemma_list
tot = '(' + '|'.join(complete) + ')'
# labels['queens_archive'] = []


# In[ ]:


dom_tabs = set(words_df.loc[words_df['lemma'].str.match('.*\[.*' + dom + '.*\]')]['id_text'])
wild_tabs = set(words_df.loc[words_df['lemma'].str.match('.*\[.*' + wild + '.*\]')]['id_text'])
dead_tabs = set(words_df.loc[words_df['lemma'].str.match('.*\[.*' + dead + '.*\]')]['id_text'])
leath_tabs = set(words_df.loc[words_df['lemma'].str.match('.*\[.*' + leath + '.*\]')]['id_text'])
prec_tabs = set(words_df.loc[words_df['lemma'].str.match('.*\[.*' + prec + '.*\]')]['id_text'])
wool_tabs = set(words_df.loc[words_df['lemma'].str.match('.*\[.*' + wool + '.*\]')]['id_text'])


# Each row of the `sparse` table below corresponds to one text, and the columns of the table correspond to the words that appear in the texts. Every cell contains the number of times a specific word appears in a certain text.

# In[ ]:


# remove lemmas that are a part of a seal as well as words that are being used to determine training classes
filter = (~words_df['label'].str.contains('s')) | words_df['lemma'].str.match('.*\[.*' + tot + '.*\]')
sparse = words_df[filter].groupby(by=['id_text', 'lemma']).count()
sparse = sparse['id_word'].unstack('lemma')
sparse = sparse.fillna(0)

#cleaning
del filter


# In[ ]:


text_length = sparse.sum(axis=1)


# If a text contains a word that is one of the designated nouns in `labels`, it is added to the set to be used for our ML model. Texts that do not contain any of these words or that contain words corresponding to more than one archive are ignored.

# In[ ]:


class_array = []

for id_text in sparse.index:
  cat = None
  number = 0
  if id_text in dom_tabs:
    number += 1
    cat = 'dom'
  if id_text in wild_tabs:
    number += 1
    cat = 'wild'
  if id_text in dead_tabs:
    number += 1
    cat = 'dead'
  if id_text in prec_tabs:
    number += 1
    cat = 'prec'
  if id_text in wool_tabs:
    number += 1
    cat = 'wool'
  if number == 1:
    class_array.append(cat)
  else:
    class_array.append(None)

class_series = pd.Series(class_array, sparse.index)


# Next we remove the texts from `sparse` that we used in the previous cell.

# In[ ]:


used_cols = []

for col in sparse.columns:
  if re.match('.*\[.*' + tot + '.*\]', col):
    used_cols.append(col)
  #elif re.match('.*PN$', col) is None:
  #  used_cols.append(col)

sparse = sparse.drop(used_cols, axis=1)


# Now the `sparse` table will be updated to contain percentages of the frequency that a word appears in the text rather than the raw number of occurrences. This will allow us to better compare frequencies across texts of different lengths.

# In[ ]:


for col in sparse.columns:
  if col != 'text_length':
    sparse[col] = sparse[col]/text_length*1000


# We must convert percentages from the previous cell into integers for the ML model to work properly.

# In[ ]:


this sparse = sparse.round()
sparse = sparse.astype(int)


# To form X, we reduce the `sparse` table to only contain texts that were designated for use above in `class_series`. Y consists of the names of the different archives.

# In[ ]:


X = sparse.loc[class_series.dropna().index]
X = X.drop(X.loc[X.sum(axis=1) == 0, :].index, axis=0)
y = class_series[X.index]


# Our data is split into a training set and a test set. The ML model first uses the training set to learn how to predict the archives for the texts. Afterwards, the test set is used to verify how well our ML model works.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                    random_state = 9)


# In[ ]:


pipe = Pipeline([
                 ('feature_reduction', SelectPercentile(score_func = f_classif)), 
                 ('weighted_multi_nb', MultinomialNB())
                 ])


# In[ ]:


from sklearn.model_selection import GridSearchCV
f = GridSearchCV(pipe, {
    'feature_reduction__percentile' : [i*10 for i in range(1, 10)],
    'weighted_multi_nb__alpha' : [i/10 for i in range(1, 10)]
    }, verbose = 0, n_jobs = -1)


# In[ ]:


f.fit(X_train, y_train);


# In[ ]:


f.best_params_


# Our best score when run on the training set is about 93.6% accuracy.

# In[ ]:


f.best_score_


# Our best score when run on the test set is very similar to above at 93.2% accuracy, which is good because it suggests that our model isn't overfitted to only work on the training set.

# In[ ]:


f.score(X_test, y_test)


# In[ ]:


predicted = f.predict(sparse)


# The `predicted_df` table is the same as the `sparse` table from above, except that we have added an extra column at the end named `prediction`. `prediction` contains our ML model's classification of which archive the text belongs to based on the frequency of the words that appear.

# In[ ]:


predicted_df = sparse.copy()
predicted_df['prediction'] = predicted


# In[ ]:


predicted_df


# In[ ]:


predicted_df.index


# ### 5.4 Testing the Model on Hand-Classified Data
# 
# Here we first use our same ML model from before on Niek's hand-classified texts from the wool archive. Testing our ML model on these tablets gives us 82.5% accuracy.

# In[ ]:


wool_hand_tabs = set(pd.read_csv('drive/MyDrive/SumerianNetworks/JupyterBook/Outputs/wool_pid.txt',header=None)[0])


# In[ ]:


hand_wool_frame = sparse.loc[wool_hand_tabs].loc[class_series.isna() == True]

f.score(X = hand_wool_frame, 
        y = pd.Series(
            index = hand_wool_frame.index, 
            data = ['wool' for i in range(0, hand_wool_frame.shape[0])] ))


# Testing our ML model on 100 random hand-classified tablets selected from among all the texts gives us 87.2% accuracy.

# In[ ]:


niek_100_random_tabs = pd.read_pickle('/content/drive/MyDrive/niek_cats').dropna()
niek_100_random_tabs = niek_100_random_tabs.set_index('pnum')['category_text']


# In[ ]:


random_frame = sparse.loc[set(niek_100_random_tabs.index)]
random_frame['result'] = niek_100_random_tabs[random_frame.index]


# In[ ]:


f.score(X=random_frame.drop(labels='result', axis=1), y = random_frame['result'])


# A large majority of the tablets are part of the domestic archive and have been classified as such.

# In[ ]:


random_frame['result'].array


# In[ ]:


f.predict(random_frame.drop(labels='result', axis=1))


# ### 5.2 Frequency Graphs
# 
# When run on all of the tablets, our ML model classifies a large portion of the texts into the domestic archive, since that is the most common one.

# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


plt.xlabel('Archive Class')
plt.ylabel('Frequency', rotation=0, labelpad=30)
plt.title('Frequencies of Predicted Archive Classes in All Tablets')
plt.xticks(rotation=45)
labels = list(set(predicted_df['prediction']))
counts = [predicted_df.loc[predicted_df['prediction'] == label].shape[0] for label in labels]
plt.bar(labels, counts);


# The below chart displays the actual frequencies of the different archives in the test set. As mentioned previously, it is visually obvious that there are many texts in the domestic archive, with comparatively very few texts in all of the other archives.

# In[ ]:


plt.xlabel('Archive Class')
plt.ylabel('Frequency', rotation=0, labelpad=30)
plt.title('Frequencies of Test Archive Classes')
plt.xticks(rotation=45)
test_counts = [(class_series[X_test.index])[class_series == label].count() for label in labels]
plt.bar(labels, np.asarray(test_counts));


# Below is a chart of the predicted frequency of the different archives by our ML model in the test set. Our predicted frequency looks very similar to the actual frequency above, which is good.

# In[ ]:


plt.xlabel('Archive Class')
plt.ylabel('Frequency', rotation=0, labelpad=30)
plt.title('Frequencies of Predicted Test Archive Classes')
plt.xticks(rotation=45)
test_pred_counts = [predicted_df.loc[X_test.index].loc[predicted_df['prediction'] == label].shape[0] for label in labels]
plt.bar(labels, np.asarray(test_pred_counts));


# Unfortunately, since our texts skew so heavily towards being part of the domestic archive, most of the other archives end up being overpredicted (i.e. our model says a text is part of that archive when it is actually not). Below we can see that the domestic archive is the only archive whose texts are not overpredicted.

# In[ ]:


plt.xlabel('Archive Class')
plt.ylabel('Rate', rotation=0, labelpad=30)
plt.title('Rate of Overprediction by Archive')
plt.xticks(rotation=45)
rate = np.asarray(test_pred_counts)/np.asarray(test_counts)*sum(test_counts)/sum(test_pred_counts)
plt.bar(labels, rate);


# ### 5.3 Accuracy By Archive
# 
# The accuracies for the dead and wild archives are relatively low. This is likely because those texts are being misclassified into the domestic archive, our largest archive, since all three of these archives deal with animals. The wool and precious archives have decent accuracies.

# In[ ]:


f.score(X_test[class_series == 'dead'], y_test[class_series == 'dead'])


# In[ ]:


f.score(X_test[class_series == 'dom'], y_test[class_series == 'dom'])


# In[ ]:


f.score(X_test[class_series == 'wild'], y_test[class_series == 'wild'])


# In[ ]:


f.score(X_test[class_series == 'wool'], y_test[class_series == 'wool'])


# In[ ]:


f.score(X_test[class_series == 'prec'], y_test[class_series == 'prec'])


# We can also look at the confusion matrix. A confusion matrix is used to evaluate the accuracy of a classification. The rows denote the actual archive, while the columns denote the predicted archive. 
# 
# Looking at the first column: 
# - 73.44% of the dead archive texts are predicted correctly
# - 1.31% of the domestic archive texts are predicted to be part of the dead archive
# - 1.47% of the wild archive texts are predicted to be part of the dead archive
# - 1.43% of the wool archive texts are predicted to be part of the dead archive
# - none of the precious archive texts are predicted to be part of the dead archive

# In[ ]:


from sklearn.metrics import confusion_matrix
archive_confusion = confusion_matrix(y_test, f.predict(X_test), normalize='true')


# In[ ]:


archive_confusion


# This is the same confusion matrix converted into real numbers of texts. Since the number of domestic archive texts is so high, even a small bit of misclassification of the domestic archive texts can overwhelm the other archives.
# 
# For example, even though only 1.3% of the domestic archive texts are predicted to be part of the dead archive, that corresponds to 43 texts, while the 73% of the dead archive texts that were predicted correctly correspond to just 47 texts. As a result, about half of the texts that were predicted to be part of the dead archive are incorrectly classified.

# In[ ]:


confusion_matrix(y_test, f.predict(X_test), normalize=None)


# ## 6 Save Results in CSV file & Pickle
