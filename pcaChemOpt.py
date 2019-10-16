import pandas as pd

# load dataset into Pandas DataFrame
df = pd.read_csv("~/Documents/Peptides/ChemOptCleaned2.csv", header=0, index_col=0)
df.head()

from sklearn.preprocessing import StandardScaler
features = df.columns
# Separating out the features
x = df.drop(['bact'], axis = 1)
# Separating out the features
#x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['bact']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
# pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
            #, columns = ['principal component 1', 'principal component 2'])
            , columns = ['principal component 1', 'principal component 2','principal component 3'])

finalDf = pd.concat([principalDf, df['bact']], axis = 1)

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['high', 'low']
colors = ['r', 'g']
for bact, color in zip(targets,colors):
    indicesToKeep = finalDf['bact'] == bact
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

pca.explained_variance_ratio_
