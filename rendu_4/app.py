import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import utile.LinearRegression as l
import utile.PolynomialRegression as p
import utile.OrdinalClassification as o
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


from sklearn import preprocessing

df = pd.read_csv('rendu_4/csv_files/housing.csv', delimiter="\s+")

df["MEDV"] = df["MEDV"] * 1000

# Calcul de la matrice de corrélation
corr_matrix = df.corr()

# Affichage sous forme de heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.savefig('rendu_4/img/matrice_correlation.png', format='png')
plt.show()


min_max_scaler = preprocessing.MinMaxScaler()

# Liste des relations à tracer
high_corr = corr_matrix[abs(corr_matrix) >= 0.7].stack().reset_index()
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']] 
high_corr = high_corr[high_corr.apply(lambda x: x['level_0'] < x['level_1'], axis=1)]


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

kf = KFold(n_splits=10, shuffle=True, random_state=42)
index = 0
for column_sels, target in relations:
    x = df.loc[:, column_sels]
    y = df[target]
    
    
    # Tracé des régressions pour chaque variable
    for col in column_sels:
        mse_mean = {}
        x_train = x[col].to_numpy()
        y_train = y.to_numpy()
        poly_reg = p.PolynomialRegression()

        # Selection du degree optimal avec validation croisée
        for degree in range(2, 6):
            mse_temp = np.array([])
            for train_index, test_index in kf.split(x_train):
                X_Train, X_Test = x_train[train_index], x_train[test_index]
                Y_Train, Y_Test = y_train[train_index], y_train[test_index]
                poly_reg.fit(X_Train, Y_Train, degree=degree)
                y_pred = poly_reg.predict(X_Test) 
                mse_temp = np.append(mse_temp, poly_reg.MSE(Y_Test, y_pred))
            mse_mean[degree] = np.mean(mse_temp)
        # Recuperation de meilleur degree
        best_degree = min(mse_mean, key=mse_mean.get)
        poly_reg.fit(x_train, y_train, degree=best_degree)

        # Génération des points de prédiction pour le graphique
        X_grid = np.arange(np.min(x_train), np.max(x_train)+0.1, 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))
        y_grid_pred = poly_reg.predict(X_grid)

        # Ajout de la regression polynomial au graphique
        axs[index].scatter(x_train, y_train, label="Données réelles", alpha=0.6) 
        axs[index].plot(X_grid, y_grid_pred, color="red", label="Régression")
        axs[index].set_title(f"{col} vs {target}")
        index += 1

# Affichage des regressions polynomiales  
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.savefig('rendu_4/img/regression_graphe.png', format='png')
plt.show()


# Affichage test Spearman
print("Tests statistiques spearman : ")
target_col_name = df.columns[-1]
for name, serie in df.items():
    if (name != target_col_name):
        p, p_value = stats.spearmanr(serie.to_numpy(), df[target_col_name].to_numpy())
        print("Spearman " + target_col_name + " x " + name + " : " + "p : " + str(p) + "  p_value : " + str(p_value))

# tranformation derniere variable en variable categorielle
name_category = target_col_name + "_category"
#df[name_category] = pd.cut(
#    df[target_col_name], 
#    bins=[-float('inf'), 20000, 35000, float('inf')], 
#    labels=[0, 1, 2]
#)

df[name_category] = pd.qcut(
    df[target_col_name], 
    q=3,
    labels=[0, 1, 2]
)
df = df.drop(columns=["MEDV"])

# Affichage test CHI2   /   les variables sont arbitrairement selectionnées ici
print("\nTests statistiques chi2 : ")
for column in ["RAD", "CHAS"]:
    contingency_table = pd.crosstab(df[column], df[name_category])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Test du chi2 {name_category} x {column} : chi2 = {chi2} ; p-value = {p}")

# Affichage test ANOVA
print("\nTests statistiques ANOVA : ")
for column in ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']:
    groupes = [df[df[name_category] == i][column] for i in df[name_category].unique()]
    f_stat, p_value = stats.f_oneway(*groupes)
    print(f"Test d'ANOVA {name_category} x {column}, F-statistique : {f_stat}, p-value : {p_value}")


# Affichage des regressions lineaires pour nouvelle variable categorielle
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
plt.savefig('rendu_4/img/regression_target_graphe.png', format='png')
plt.show()



X = df.iloc[:, :-1].to_numpy()
X = np.c_[np.ones(X.shape[0]), X]
y = df.iloc[:, -1].to_numpy()

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

accuracy_scores = []
all_y_true = []
all_y_pred = []
# Cross validation 
for train_index, test_index in kf.split(X_pca, y):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]

    log_reg = o.OrdinalClassification(3)
    log_reg.fit(X_train, y_train, alpha=0.1)
    
    y_pred = log_reg.predict(X_test)

    # Stocker les résultats pour la moyenne
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)


# Affichage de la moyenne des scores d'accuracy sur les folds
print(f'\nRegression logistique, Accuracy moyenne (Validation croisée) : {np.mean(accuracy_scores):.4f}')
# Affichage du rapport de classification
print("Rapport de Classification :\n", classification_report(all_y_true, all_y_pred))

log_reg.fit(X_pca, y, alpha=0.1)


# Visualisation des deux premières composantes principales
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection PCA")
plt.colorbar(label='MEDV_category')
plt.show()



## Calcules et affichage des courbes ROC des différents modèles 
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))
axs = axs.flatten()
index = 0

# Transformation des étiquettes en un format binaire pour chaque classe
y_bin = label_binarize(y, classes=[0, 1, 2])

# Modele de classification ordinal
fpr = {}
tpr = {}
roc_auc = {}

for i in range(3):  # 3 classes
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], log_reg.proba(X_pca, i))
    roc_auc[i] = auc(fpr[i], tpr[i])
for i in range(3):
    axs[index].plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')
axs[index].plot([0, 1], [0, 1], color='gray', linestyle='--')
axs[index].set_title(f"Classification Ordinale")
axs[index].set_xlabel('Taux de Faux Positifs (FPR)')
axs[index].set_ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC Régression Logistique Ordinal')
plt.legend(loc='lower right')

index += 1

# Selection des différents modèles
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "SVM": SVC(probability=True, kernel='rbf')
}

# Entraînement et évaluation
results = {}

for name, model in models.items():
    model.fit(X_pca, y)
    y_pred = model.predict(X_pca)
    y_proba = model.predict_proba(X_pca)
    accuracy = accuracy_score(y, y_pred)
    print(f"\n{name} - Accuracy: {accuracy:.4f}")
    print(classification_report(y, y_pred))

    # Courbe ROC
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(3):
        axs[index].plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')
    axs[index].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axs[index].set_title(f"{name}")
    axs[index].set_xlabel('Taux de Faux Positifs (FPR)')
    axs[index].set_ylabel('Taux de Vrais Positifs (TPR)')
    index += 1
    results[name] = accuracy


plt.title("Courbes ROC des modèles")
plt.savefig('rendu_4/img/ROC_graphe.png', format='png')
plt.show()
