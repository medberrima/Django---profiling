from django.shortcuts import render



import pandas as pd
import os
import string
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import ast
from sklearn.preprocessing import LabelEncoder 


# In[2]:


def supprimer_lignes_utilisateur_repete(dataframe):
    # Supprimer les lignes répétées en conservant uniquement la première occurrence de chaque identifiant d'utilisateur
    dataframe_unique = dataframe.drop_duplicates(subset='User id', keep='first')
    
    # Retourner le DataFrame sans les lignes répétées
    return dataframe_unique


# In[3]:


def extraire_lignes_utilisateur_repete(dataframe):
    # Trouver les identifiants d'utilisateurs répétés
    utilisateurs_repetes = dataframe[dataframe.duplicated(subset='User id', keep=False)]['User id']
    
    # Extraire les lignes où se trouve l'utilisateur répété
    lignes_utilisateur_repete = dataframe[dataframe['User id'].isin(utilisateurs_repetes)]
    
    # Retourner les lignes extraites
    return lignes_utilisateur_repete


# In[4]:


def lecture_fichier(filename):
    path = filename
    if filename.lower().endswith('.csv'):
        dataframe = pd.read_csv(path)
        dataframe=dataframe.drop(['Created at','Updated at', 'S.No'], axis=1)
        dataframe=supprimer_lignes_utilisateur_repete(dataframe)
    return dataframe


# In[5]:


def first_df():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    queney_path = os.path.join(current_dir, 'queney.csv')
    servir_path = os.path.join(current_dir, 'servir.csv')
    profil_path = os.path.join(current_dir, 'profil.csv')

    queney = lecture_fichier(queney_path)
    servir = lecture_fichier(servir_path)
    profil = lecture_fichier(profil_path)
    df = pd.merge(servir, queney, on='User id', suffixes=('', ' '))
    df = pd.merge(df, profil, on='User id', suffixes=('', '  '))
    df.insert(0, 'User id', df.pop('User id'))
    return df


# In[6]:


def merge(df, filename):
    df = pd.merge(df, lecture_fichier(filename), on='User id', suffixes=('', '   '))
    df.insert(0, 'User id', df.pop('User id'))
    return df


# In[7]:


def lecture_dictionnaire(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    dictionnaire = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        dictionnaire = ast.literal_eval(content)
    return dictionnaire


# In[8]:


def normaliser(df):
    # Télécharger les ensembles de données nécessaires
    nltk.download('stopwords')
    nltk.download('wordnet')


    # Normaliser les noms de colonnes en minuscules et enlever la ponctuation
    df = df.applymap(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)) if type(x) == str else x)
    # Définir les stopwords et initialiser le lemmatizer
    stop_words = set(stopwords.words('french'))
    lemmatizer = WordNetLemmatizer()

    # Fonction pour lemmatiser les mots et enlever les stopwords
    def preprocess(text):
        if isinstance(text, str):
            words = text.split()
            words = [lemmatizer.lemmatize(word, pos='n') for word in words if word not in stop_words]
            return ' '.join(words)
        else:
            return text
    # Appliquer la fonction de prétraitement sur les noms de colonnes
    df = df.applymap(preprocess)
    return df


# In[9]:


def compatible_columns(row, synonym_dict):
    compatible_columns = {}
    for col in row.index:
        value = row[col]
        if isinstance(value, str):  # Vérifier si la valeur est une chaîne de caractères
            words = value.split()
            for critere, details in synonym_dict.items():
                similarite = details.get('similarite', [])
                synonymes = details.get('synonymes', {})
                if any(synonyme in words for synonyme in similarite):
                    if critere not in compatible_columns:
                        compatible_columns[critere] = [col]
                    else:
                        compatible_columns[critere].append(col)
                else:
                    for synonyme, synonyme_values in synonymes.items():
                        if any(syn in words for syn in synonyme_values):
                            if critere not in compatible_columns:
                                compatible_columns[critere] = [col]
                            else:
                                compatible_columns[critere].append(col)

    if compatible_columns:
        return True, compatible_columns

    return False, None


# In[10]:


def compatible_columns_lignes(df):
    for i in range(df.shape[0]):
        result = compatible_columns(df.iloc[i], lecture_dictionnaire('dic.txt'))
        if result is not None and result[0] == True:
            return result[0], result[1]
    # Handle the case when no compatible value is found
    return False, None


# In[11]:


def extraire_colonnes_club(df):
    colonnes_club = []
    
    # Recherche des colonnes contenant 'club' dans leur nom
    for colonne in df.columns:
        if 'club' in colonne.lower():
            colonnes_club.append(colonne)
    
    # Recherche des colonnes contenant 'club' dans certaines lignes
    for colonne in df.columns:
        for valeur in df[colonne]:
            if isinstance(valeur, str) and 'club' in valeur.lower() and colonne not in colonnes_club:
                colonnes_club.append(colonne)
                break
    
    # Création d'un nouveau DataFrame avec les colonnes extraites
    df_club = df[colonnes_club]
    
    return df_club


# In[12]:


def compatible_columns_colonnes(df, synonym_dict):
    compatible_columns = {}
    for col in df.columns:
        value = str(col)
        words = value.split()
        for critere, details in synonym_dict.items():
            similarite = details["similarite"]
            synonymes = details["synonymes"]
            if any(synonyme in words for synonyme in similarite):
                if critere not in compatible_columns:
                    compatible_columns[critere] = [col]
                else:
                    compatible_columns[critere].append(col)
            for syn, syn_values in synonymes.items():
                if any(syn_value in words for syn_value in syn_values):
                    if critere not in compatible_columns:
                        compatible_columns[critere] = [col]
                    else:
                        compatible_columns[critere].append(col)

    if compatible_columns:
        return compatible_columns

    return None


# In[13]:


def creer_liste_reponses_ligne(row, criteres):
    reponses = []
    liste_reponses = []

    for colonne in row.index:
        if colonne.endswith('Qualification'):
            liste_reponses.append(reponses)
            reponses = []
            continue

        if colonne != 'User id' and colonne not in criteres:
            reponses.append(row[colonne])

    liste_reponses.append(reponses)

    return liste_reponses


# In[14]:


def rechercher_sous_ensemble(dictionnaire_synonymes, reponse):
    for critere, valeurs in dictionnaire_synonymes.items():
        synonymes = valeurs.get('synonymes', {})
        for sous_ensemble, liste_synonymes in synonymes.items():
            if reponse in liste_synonymes:
                return sous_ensemble
    return None


# In[15]:


def verifier_coherence(reponses, question_colonne, dictionnaire_synonymes):
    for i in range(1, len(reponses)):
        current_reponse = reponses[i]
        previous_reponse = reponses[i - 1]

        ensemble_synonymes_current = rechercher_sous_ensemble(dictionnaire_synonymes, current_reponse)
        ensemble_synonymes_previous = rechercher_sous_ensemble(dictionnaire_synonymes, previous_reponse)

        if ensemble_synonymes_current is None or ensemble_synonymes_previous is None or ensemble_synonymes_current != ensemble_synonymes_previous:
            return False

    return True


# In[16]:


from collections import Counter

def mots_repetes(liste):
    mots_repetes = []
    mots_comptes = Counter()

    for element in liste:
        mots = element.split()
        mots_comptes.update(mots)

    for mot, count in mots_comptes.items():
        if count > 1:
            mots_repetes.append(mot)

    return mots_repetes


# In[17]:


def remplir(df, dictionnaire_synonymes):
    for index, row in df.iterrows():
        reponses = creer_liste_reponses_ligne(row, dictionnaire_synonymes.keys())
        nbre_col = 0
        for reponse in reponses:
            for colonne in df.columns[nbre_col + 1:]:
                nbre_col += 1
                if colonne.endswith('Qualification'):
                    qualification_colonne = colonne
                    question_colonne = colonne[:-14]
                    if question_colonne=='Sport':
                        if len(mots_repetes(reponse))!=0:
                            df.at[index, qualification_colonne] = 'Cohérent'
                            df.at[index, question_colonne] = ", ".join(mots_repetes(reponse))
                        else:
                            df.at[index, qualification_colonne] = 'Incohérent'
                            df.at[index, question_colonne] = None
                        break
                    elif question_colonne in ['Permis de conduire', 'Motorisé'] :
                        is_coherent = verifier_coherence(reponse, question_colonne, dictionnaire_synonymes)
                        if (is_coherent or all(element == reponse[0] for element in reponse)) and (row['Age'] == 'moins 18 an'):
                            df.at[index, qualification_colonne] = 'Incohérent'
                            dff.at[index, question_colonne] = None
                        elif is_coherent or all(element == reponse[0] for element in reponse):
                            df.at[index, qualification_colonne] = 'Cohérent'
                            df.at[index, question_colonne] = reponse[0]
                        else:
                            df.at[index, qualification_colonne] = 'Incohérent'
                            df.at[index, question_colonne] = None
                        break
                    else:
                        is_coherent = verifier_coherence(reponse, question_colonne, dictionnaire_synonymes)
                        if is_coherent or all(element == reponse[0] for element in reponse):
                            df.at[index, qualification_colonne] = 'Cohérent'
                            df.at[index, question_colonne] = reponse[0]
                        else:
                            df.at[index, qualification_colonne] = 'Incohérent'
                            df.at[index, question_colonne] = None
                        break
            continue

    return df


# In[18]:


def get_common_columns(df):
    d = df
    data = d[['User id']]
    for j in range(1, d.shape[1]):
        c = d.columns[j]
        for i in range(j+1, d.shape[1]):
            if c.rstrip() == d.columns[i].rstrip():
                data[c] = d[c]
                data[d.columns[i]] = d[d.columns[i]]
        if c in data.columns:
            break  # Arrêter la boucle externe si une colonne commune est trouvée
    d = d.drop(data.drop('User id', axis=1).columns, axis=1)
    return data, d


# In[19]:


def remplir_common_columns(df):
    columns = df.columns[1:]
    similar_counts = df[columns].eq(df[columns].shift(axis=1)).sum(axis=1)
    
    def fill_value(row):
        if mots_repetes(row):
            return ", ".join(mots_repetes(row))
        elif similar_counts[row.name] >= len(columns) / 2:
            return row[1]
        else:
            return None

    df['Valeur'] = df.apply(fill_value, axis=1)
    
    return df


# In[20]:


def trouver_reponses_possibles(colonne):
    reponses_possibles = {}
    
    reponses_colonne = colonne.dropna().unique().tolist()
    reponses_possibles= reponses_colonne
    
    return sorted(reponses_possibles)


# In[21]:


df = normaliser(first_df())
df


# In[22]:


dataframe=df[['User id']]
dataframe


# In[23]:


clubs=extraire_colonnes_club(df)
clubs.insert(0, 'User id', df['User id'])
df = df.drop(clubs.drop('User id', axis=1).columns, axis=1)
clubs=remplir_common_columns(clubs)


# In[24]:


clubs


# In[25]:


dataframe = pd.merge(dataframe, clubs[['User id', clubs.columns[3]]], on='User id').rename(columns={clubs.columns[3]: 'Club'})
dataframe


# In[26]:


n, liste1=compatible_columns_lignes(df)


# In[27]:


liste1


# In[28]:


liste2=compatible_columns_colonnes(df, lecture_dictionnaire('dic.txt'))
liste2


# In[29]:


dff = df[['User id']]  # Sélection de la colonne 'User id' du DataFrame d'origine

# Ajout de chaque clé en tant que colonne avec les valeurs correspondantes
for key, values in liste1.items():
    if key not in dff.columns:
        for value in values:
            dff[value] = df[value]
        dff[key + ' Qualification'] = 'Qualification'
        dff[key] = key

for key, values in liste2.items():
    if key in liste1.keys():
        for value in values:
            if value not in dff.columns:
                dff.insert(dff.columns.get_loc(key) - 1, value, df[value])
    else:
        for value in values:
            if value not in dff.columns:
                dff[value] = df[value]
        dff[key + ' Qualification'] = 'Qualification'
        dff[key] = key


# In[30]:


# Exécution
dff = remplir(dff, lecture_dictionnaire('dic.txt'))
dff


# In[31]:


for key in liste1.keys():
    dataframe = pd.merge(dataframe, dff[['User id', key]], on='User id')
for key in liste2.keys():
    if key not in dataframe.columns:
        dataframe = pd.merge(dataframe, dff[['User id', key]], on='User id')


# In[32]:


for cols in liste1.values():
    for col in cols:
        if col in df.columns:
            df=df.drop(col, axis=1)


# In[33]:


for cols in liste2.values():
    for col in cols:
        if col in df.columns:
            df=df.drop(col, axis=1)


# In[34]:


data, df = get_common_columns(df)
while data.shape[1] > 1:
    rt = remplir_common_columns(data)
    df[rt.columns[1]] = rt[rt.columns[-1]]
    data, df = get_common_columns(df)
 


# In[35]:


df_final=pd.merge(dataframe, df, on='User id')
df_final


# In[36]:


dataframe


# In[37]:


df_final.to_csv('dffinal.csv')


# In[38]:


for i in range(1,df_final.shape[1]):
    print(i, df_final.columns[i])

label_encoder = LabelEncoder()
X = pd.DataFrame(df_final.iloc[:, 0])
t = int(input())
n = 0
while t in range(df_final.shape[1]):
    n += 1
    column = df_final.columns[t]
    found_in_keywords = False
    for key in lecture_dictionnaire('dic.txt').keys():
        if column == key:
            found_in_keywords = True
            break
    
    if not found_in_keywords:
        nom = input('Donner un nom à cette colonne : ')
    else:
        nom = column
    X.insert(n, nom, label_encoder.fit_transform(df_final.iloc[:, t]))
    t = int(input())


# In[39]:


X


# In[40]:


import numpy as np
from scipy.stats import gaussian_kde
# Estimer la densité des données avec une estimation de Silverman
data = X.drop('User id', axis=1).values
density = gaussian_kde(data.T)

# Calculer la bande passante avec la règle de Silverman
n = data.shape[1]  # Nombre de dimensions
n_samples = data.shape[0]  # Nombre d'échantillons
silverman_factor = (n + 2) / (4 * (n + 1))
silverman_bandwidth = np.power(n_samples * silverman_factor, -1 / (n + 4))

# Afficher la valeur de la bande passante estimée
print(f"Bande passante estimée: {silverman_bandwidth}")


# In[41]:


from sklearn.cluster import MeanShift
# Créer l'instance du modèle Mean Shift
ms = MeanShift(bandwidth=silverman_bandwidth)
# Ajuster le modèle sur les données 
ms.fit(data)
# Obtenir les labels de cluster pour chaque point de données
labels = ms.labels_

# Afficher le nombre de clusters trouvés
n_clusters = len(set(labels))
print(f'Nombre de clusters : {n_clusters}')


# In[42]:


df_final['cluster'] = ms.labels_
df_final


# In[43]:


X['cluster'] = ms.labels_
X


# In[44]:


df_final.describe()


# In[45]:


#Mieux comprendre les clusters
#Analyse des caractéristiques des clusters
cluster_stats = X.groupby("cluster").describe()
cluster_stats


# In[46]:


import matplotlib.pyplot as plt    
#Analyse2
# Effectuer l'analyse des clusters
clusters = X['cluster'].unique()  # Remplacez 'cluster_column' par le nom de la colonne contenant les numéros de cluster

# Analyser les caractéristiques des clusters
for cluster in clusters:
    cluster_data = X[X['cluster'] == cluster]
    cluster_mean = cluster_data.mean()
    cluster_std = cluster_data.std()
    # Afficher les caractéristiques du cluster
    print(f"Cluster {cluster} - Moyenne : {cluster_mean}, Écart-type : {cluster_std}")
    # Visualiser les distributions des données dans chaque cluster
    cluster_data.hist()  # Vous pouvez personnaliser les paramètres du graphique selon vos besoins
    plt.title(f"Distribution des données - Cluster {cluster}")
    plt.show()


# In[47]:


#Pour analyser les caractéristiques spécifiques des données qui ont conduit à la formation des clusters et identifier les variables ou les combinaisons de variables influentes, vous pouvez utiliser des techniques d'analyse exploratoire des données telles que l'analyse en composantes principales (PCA) ou l'analyse discriminante linéaire (LDA). Voici un exemple de code en utilisant la bibliothèque scikit-learn pour effectuer cette analyse :
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler


# Analyser les caractéristiques spécifiques des données pour chaque cluster
cluster_means = X.groupby('cluster').mean()

# Identifier les variables ayant une influence significative sur la formation des clusters
significant_vars = cluster_means.columns

cluster_means
print(significant_vars, cluster_means)


# In[48]:


#des tests supplémentaires pour évaluer la qualité des clusters obtenus à l'aide de techniques d'apprentissage automatique supervisées :
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score


X=X.drop('User id', axis=1)
# Diviser les données en variables indépendantes (X) et variable cible (y)
x = X.drop(['cluster'], axis=1)
y = X['cluster']

# Diviser les données en ensembles d'apprentissage et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Appliquer une technique d'apprentissage automatique supervisée (par exemple, Régression Logistique)
model = LogisticRegression()
model.fit(x_train, y_train)

# Prédire les clusters pour les données de test
y_pred = model.predict(x_test)

# Évaluer la qualité des clusters en utilisant la métrique de silhouette
silhouette = silhouette_score(x, X['cluster'])
print("Score de silhouette : ", silhouette)


# In[49]:


# Afficher les centres des clusters
cluster_centers = ms.cluster_centers_
print("Centres des clusters :")
cluster_centers


# In[50]:


import matplotlib.pyplot as plt    

# Obtenir les coordonnées des centroids des clusters
centroids = ms.cluster_centers_

# Tracer le graphique
for cluster_label in np.unique(X['cluster']):
    cluster_data = X[X['cluster'] == cluster_label]
    plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'cluster {cluster_label}')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=100)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('clusters obtenus avec MeanShift')
plt.legend()
plt.show()


# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler

def visualiser_clusters(X, cluster_labels):
    # Prétraitement des données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    # Appliquer l'algorithme MeanShift
    meanshift = MeanShift()
    meanshift.fit(scaled_data)

    # Assigner chaque instance à un cluster
    cluster_labels['cluster'] = meanshift.labels_

    # Obtenir les coordonnées des centroids des clusters
    centroids = meanshift.cluster_centers_

    # Tracer le graphique
    num_cols = X.shape[1]
    fig, axes = plt.subplots(nrows=num_cols, ncols=num_cols, figsize=(10, 10))
    fig.tight_layout(pad=2.0)

    for i in range(num_cols):
        for j in range(num_cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(X.iloc[:, i], bins=10, color='lightblue', edgecolor='black')
                ax.set_xlabel(f'Variable {i + 1}')
                ax.set_ylabel('Count')
            else:
                for cluster_label in np.unique(cluster_labels['cluster']):
                    cluster_data = cluster_labels[cluster_labels['cluster'] == cluster_label]
                    ax.scatter(X.iloc[:, j], X.iloc[:, i], c=cluster_data['cluster'], cmap='viridis')
                ax.scatter(centroids[:, j], centroids[:, i], marker='X', color='red', s=100)
                ax.set_xlabel(f'Variable {j + 1}')
                ax.set_ylabel(f'Variable {i + 1}')

    plt.subplots_adjust(top=0.92)
    plt.suptitle('Clusters obtenus avec MeanShift')
    plt.show()

cluster_labels=[]
for i in np.unique(X['cluster']):
    cluster_labels = X[X['cluster']]
visualiser_clusters(X, cluster_labels)


# In[ ]:




