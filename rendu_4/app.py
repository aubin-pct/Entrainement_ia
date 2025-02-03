import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import utile.LinearRegression as l

from sklearn import preprocessing

df = pd.read_csv('rendu_4/csv_files/housing.csv', delimiter="\s+")

df["MEDV"] = df["MEDV"] * 1000
# Calcul de la matrice de corrélation

corr_matrix = df.corr()

# Affichage sous forme de heatmap=
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.savefig('rendu_4/img/matrice_correlation.jpg', format='jpg')
plt.show()


min_max_scaler = preprocessing.MinMaxScaler()
# Liste des relations à tracer

high_corr = corr_matrix[abs(corr_matrix) >= 0.7].stack().reset_index()
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
relations.append((tab, temp))

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


# Affichage stat Spearman
target_col_name = df.columns[-1]
for name, serie in df.items():
    if (name != target_col_name):
        p, p_value = stats.spearmanr(serie.to_numpy(), df[target_col_name].to_numpy())
        print("Spearman " + target_col_name + " x " + name + " : " + "p : " + str(p) + "  p_value : " + str(p_value))

# tranformation derniere variable en variable categorielle
name_category = target_col_name + "_category"
df[name_category] = pd.cut(
    df[target_col_name], 
    bins=[-float('inf'), 20000, 40000, float('inf')], 
    labels=[0, 1, 2]
)

# Affichage regression lineaire pour nouvelle variable categorielle
fig, axs = plt.subplots(ncols=4, nrows=int(np.ceil(len(df.columns) / 4)) , figsize=(20, 10))
axs = axs.flatten()
index = 0
lin_reg = l.LinearRegression()
for name, serie in df.items():
    if (name != target_col_name and name != name_category):
        lin_reg.fit(serie.to_numpy(), df[name_category].to_numpy())
        y_pred = lin_reg.predict(serie.to_numpy())
        axs[index].scatter(serie, df[name_category].to_numpy(), label="Données réelles", alpha=0.6)  # Points
        axs[index].plot(serie, y_pred, color="red", label="Régression")  # Ligne
        axs[index].set_title(f"{name} vs {name_category}")
        axs[index].legend()
        index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

