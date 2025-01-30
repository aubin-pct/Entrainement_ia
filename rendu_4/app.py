import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing

df = pd.read_csv('rendu_4/csv_files/housing.csv', delimiter="\s+")

# Calcul de la matrice de corrélation
corr_matrix = df.corr()

# Affichage sous forme de heatmap=
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.savefig('rendu_4/img/matrice_correlation.jpg', format='jpg')
plt.show()

min_max_scaler = preprocessing.MinMaxScaler()
# Liste des relations à tracer

high_corr = corr_matrix[abs(corr_matrix) > 0.7].stack().reset_index()
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']] 
high_corr = high_corr[high_corr.apply(lambda x: x['level_0'] < x['level_1'], axis=1)]

print(high_corr)

relations = []
temp = None
tab = []
for _, serie in high_corr.iterrows():
    if (temp == None):
        temp = serie["level_0"]
    elif (temp != serie["level_0"]):
        relations.append((tab, temp))
        temp = serie["level_0"]
        tab = []
    tab.append(serie["level_1"])

nb_row = len(high_corr) // 4
# Création de la figure avec des sous-graphiques
fig, axs = plt.subplots(ncols=4, nrows=nb_row, figsize=(20, 10))
axs = axs.flatten()

index = 0
for column_sels, target in relations:
    x = df.loc[:, column_sels]
    y = df[target]
    
    # Normalisation
    x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
    
    # Tracé des régressions pour chaque variable
    for col in column_sels:
        sns.regplot(y=y, x=x[col], ax=axs[index])
        axs[index].set_title(f"{col} vs {target}")
        index += 1


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.savefig('rendu_4/img/regression_graphe.jpg', format='jpg')
plt.show()
