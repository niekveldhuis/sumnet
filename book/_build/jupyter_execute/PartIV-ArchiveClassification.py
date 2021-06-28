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


words_df = pd.read_csv('output/part_3_words_output.csv')
#words_df = pd.read_pickle('output/part_3_words_output.p')


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


known.to_csv('output/part_4_known.csv')
known.to_pickle('output/part_4_known.p')
unknown.to_csv('output/part_4_unknown.csv')
unknown.to_pickle('output/part_4_unknown.p')
unknown_0.to_csv('output/part_4_unknown_0.csv')
unknown_0.to_pickle('output/part_4_unknown_0.p')


# In[ ]:


#known = pd.read_pickle('part_4_known.p')
#unknown = pd.read_pickle('part_4_unknown.p')
#unknown_0 = pd.read_pickle('part_4_unknown_0.p')

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
# Here we will train our model using a Naive Bayes Model to predict archives based on the features made in the previous subsection. Here, we make the assumption that the features are independent of eachother, from which the descriptor _naive_ comes from. So:
# 
# $$P(x_i|y; x_1, x_2, ..., x_{i-1}, x_{i+1}, ..., x_n) = P(x_i| y)$$
# 
# and:
# 
# $$P(x_1, x_2, ..., x_n | y) = \prod_{i=1}^{n} P(x_i | y)$$
# 
# Moreover we will be using a Bayesian probability:
# 
# $$P(y|x_1, x_2, ..., x_n) = \frac{P(y)P(x_1, x_2, ..., x_n | y)}{P(x_1, x_2, ..., x_n)}$$
# 
# eg. the probability of a particular tablet (defined by features $x_1, x_2, ..., x_n$) is in archive $y$, is equal to the probability of getting a tablet from archive $y$ times the probability you would get a particular set of features $x_1, x_2, ..., x_n$ divided by the probability of getting a particular set of features $x_1, x_2, ..., x_n$.
# 
# Thus we can simpify this to:
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


# ### 2.5 Random Forest

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


# ## 5 Save Results in CSV file & Pickle
