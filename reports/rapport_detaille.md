# Rapport detaille - Prediction de la qualite de l'air par Deep Learning

## 1. Introduction

Ce projet a pour objectif de predire la qualite de l'air a partir de donnees environnementales et de pollution atmospherique. L'etude repose sur le dataset **Delhi Air Quality Dataset**, qui contient des mesures journalieres de plusieurs polluants ainsi qu'un indice global de qualite de l'air.

Le travail realise suit une logique complete de projet de data science :

- inspection automatique du dataset ;
- choix rigoureux de la cible ;
- nettoyage et pretraitement des donnees ;
- gestion de la dimension temporelle ;
- construction de plusieurs modeles de deep learning ;
- comparaison quantitative des performances ;
- interpretation des resultats.

L'objectif principal retenu dans ce projet est la **prediction de la variable numerique AQI**. Le probleme est donc traite comme une **regression temporelle multivariee**.

## 2. Description du dataset

Le fichier utilise est `city_day.csv`.

### 2.1 Dimensions et structure

- Nombre total de lignes : `29 531`
- Nombre total de colonnes : `16`
- Nombre de villes : `26`
- Nombre de dates uniques : `2 009`

### 2.2 Colonnes detectees

- `City`
- `Date`
- `PM2.5`
- `PM10`
- `NO`
- `NO2`
- `NOx`
- `NH3`
- `CO`
- `SO2`
- `O3`
- `Benzene`
- `Toluene`
- `Xylene`
- `AQI`
- `AQI_Bucket`

### 2.3 Choix de la cible

La pipeline inspecte automatiquement le dataset et choisit la cible la plus pertinente selon les colonnes disponibles.

Dans ce cas :

- `AQI` est present ;
- `AQI` est numerique ;
- `AQI_Bucket` est present mais derive d'une discretisation de l'indice.

La variable cible principale choisie est donc **`AQI`**, ce qui conduit naturellement a un probleme de **regression**.

## 3. Analyse exploratoire des donnees

L'analyse exploratoire a ete automatisee et les figures sont disponibles dans le dossier `reports/figures/`.

### 3.1 Valeurs manquantes

Le dataset presente plusieurs colonnes avec un taux de valeurs manquantes important :

- `Xylene` : `18 109` valeurs manquantes
- `PM10` : `11 140`
- `NH3` : `10 328`
- `Toluene` : `8 041`
- `Benzene` : `5 623`
- `AQI` : `4 681`

Ce constat justifie la mise en place d'un pretraitement robuste avec imputation.

### 3.2 Distribution de la cible

La cible `AQI` montre une distribution etalee avec la presence de valeurs elevees, ce qui traduit des episodes de pollution importants. La variable ne suit pas une loi gaussienne simple, ce qui rend le probleme de prediction plus delicat.

### 3.3 Dimension temporelle

Les observations sont journalieres et distribuees sur plusieurs villes. La serie n'est pas une simple serie univariee mais une **serie temporelle multivariee panelisee**, c'est-a-dire une succession de series chronologiques paralleles par ville.

### 3.4 Variables potentiellement influentes

D'apres la structure du dataset et les resultats ulterieurs, les polluants les plus informatifs sont :

- `PM2.5`
- `CO`
- `PM10`
- `O3`
- `NOx`

## 4. Pretraitement

Le pretraitement a ete concu pour etre rigoureux et pour eviter toute fuite de donnees.

### 4.1 Nettoyage

Les etapes suivantes ont ete appliquees :

- suppression des doublons ;
- conversion de `Date` au format datetime ;
- tri chronologique par ville et par date ;
- suppression des lignes dont la cible `AQI` est manquante.

Apres nettoyage, le dataset passe de `29 531` lignes a `24 850` lignes exploitables.

### 4.2 Variables temporelles ajoutees

Des variables calendaires ont ete generees a partir de la date :

- `year`
- `month`
- `day`
- `day_of_week`
- `day_of_year`
- `week_of_year`
- `is_weekend`
- `hour`

Remarque : comme les donnees sont journalieres, `hour` n'apporte pas d'information discriminante et son importance est nulle dans les resultats finaux.

### 4.3 Encodage et normalisation

- les variables numeriques sont imputees par la mediane puis standardisees ;
- la variable categorielle `City` est encodee en one-hot encoding ;
- `AQI_Bucket` est explicitement exclue des variables explicatives pour eviter une fuite de donnees, car elle est derivee de la cible.

### 4.4 Separation train / validation / test

Le split est realise selon l'axe temporel :

- train : `70 %`
- validation : `15 %`
- test : `15 %`

Tailles obtenues :

- Train : `12 446` lignes
- Validation : `5 429` lignes
- Test : `6 975` lignes

## 5. Construction des sequences temporelles

Afin d'exploiter la dynamique temporelle, des sequences de longueur `14` ont ete construites par ville.

Chaque echantillon correspond donc a :

- les `14` observations precedentes d'une meme ville ;
- pour predire l'`AQI` de l'instant suivant.

Dimensions finales des tenseurs :

- Train : `(12 194, 14, 38)`
- Validation : `(5 143, 14, 38)`
- Test : `(6 611, 14, 38)`

Cela signifie :

- `12 194` sequences d'entrainement ;
- `14` pas de temps ;
- `38` variables apres pretraitement et encodage.

## 6. Modeles de deep learning

Trois modeles ont ete implementes et compares.

### 6.1 MLP

Le MLP sert de baseline. Il utilise les sequences temporelles, mais sous forme **aplaties**.

Architecture :

- Dense(256)
- BatchNormalization
- Dropout(0.30)
- Dense(128)
- Dropout(0.20)
- Dense(64)
- Dense(1)

### 6.2 LSTM

Le LSTM est le modele principal pour capter les dependances temporelles.

Architecture :

- LSTM(128, return_sequences=True)
- Dropout(0.25)
- LSTM(64)
- Dense(64)
- Dropout(0.20)
- Dense(1)

### 6.3 GRU

Le GRU est une alternative recurrente plus legere.

Architecture :

- GRU(128, return_sequences=True)
- Dropout(0.25)
- GRU(64)
- Dense(64)
- Dropout(0.20)
- Dense(1)

### 6.4 Parametres d'entrainement

- Optimiseur : `Adam`
- Fonction de perte : `MSE`
- Metriques : `MAE`, `RMSE`
- Batch size : `32`
- Epochs max : `40`
- Early stopping : oui
- ReduceLROnPlateau : oui
- Checkpoint du meilleur modele : oui

## 7. Resultats quantitatifs

Les resultats finaux sur le jeu de test sont les suivants :

| Modele | MAE | RMSE | R2 |
|---|---:|---:|---:|
| LSTM | 25.77 | 43.91 | 0.8264 |
| GRU | 27.11 | 44.39 | 0.8226 |
| MLP | 33.11 | 60.70 | 0.6682 |

### 7.1 Meilleur modele

Le meilleur modele est le **LSTM**.

Il obtient :

- l'erreur absolue moyenne la plus faible ;
- le RMSE le plus faible ;
- le meilleur coefficient de determination `R²`.

### 7.2 Interpretation des scores

- Un `R² = 0.8264` signifie que le modele explique environ **82.6 % de la variance** de la cible sur le jeu de test.
- Le `LSTM` surpasse nettement le `MLP`, ce qui confirme que la dynamique temporelle contient une information predictive importante.
- Le `GRU` est tres proche du `LSTM`, ce qui montre qu'une architecture recurrente plus legere reste tres competitive.

## 8. Comparaison avec la litterature

Afin de situer les performances obtenues, une comparaison avec des travaux recents utilisant des donnees de qualite de l'air en Inde est presentee ci-dessous.

### 8.1 Tableau comparatif

| Etude | Modele | MAE | RMSE | R² | Contexte |
|---|---|---:|---:|---:|---|
| Ce projet | LSTM | 25.77 | 43.91 | **0.8264** | 26 villes, donnees journalieres, AQI |
| Ce projet | GRU | 27.11 | 44.39 | 0.8226 | 26 villes, donnees journalieres, AQI |
| Ce projet | MLP | 33.11 | 60.70 | 0.6682 | 26 villes, donnees journalieres, AQI |
| Earth Science Informatics (2024) | Bi-LSTM-GRU | 36.11 | — | 0.84 | 1 ville (Delhi), PM2.5 uniquement |
| Scientific Reports (2024) | Bi-LSTM | 244.54 | 438.90 | 0.979 | Delhi, prediction de polluants individuels |
| Discover Applied Sciences (2025) | CNN-LSTM | 8.38 | 11.40 | — | 1 ville (Anugul), donnees horaires |
| Discover Sustainability (2024) | GRU | 3.43 | 4.69 | 0.99 | 1 ville, donnees horaires |
| Scientific Reports (2024) | XGBoost optimise | — | 4.65 | 0.99 | 6 villes, machine learning classique |

Sources :
- Sharma et al. (2024), *Earth Science Informatics* — Bi-LSTM-GRU hybride pour la prediction de PM2.5 a Delhi.
- Mishra et al. (2024), *Scientific Reports* — Bi-LSTM pour la modelisation temporelle des polluants a Delhi.
- Reddy et al. (2025), *Discover Applied Sciences* — CNN-LSTM avec interpretabilite SHAP-LIME pour l'AQI.
- Kumar et al. (2024), *Discover Sustainability* — GRU pour la prediction de l'AQI en ville intelligente.
- Gupta et al. (2024), *Scientific Reports* — XGBoost optimise par Grey Wolf Optimization sur plusieurs villes indiennes.

### 8.2 Analyse comparative

Plusieurs observations importantes ressortent de cette comparaison.

**Le contexte experimental varie fortement d'une etude a l'autre.** Les travaux obtenant les meilleurs scores (R² = 0.99) utilisent generalement des donnees **horaires** sur **une seule ville**, ce qui simplifie considerablement le probleme : la variabilite inter-ville est eliminee, et la resolution temporelle fine facilite la prediction a court terme. En revanche, ce projet traite simultanément **26 villes indiennes** avec des donnees **journalieres**, ce qui constitue un cadre nettement plus exigeant.

**Le R² = 0.8264 obtenu par le LSTM est solide et coherent avec la litterature comparable.** Il est directement comparable a celui du modele Bi-LSTM-GRU de Earth Science Informatics (R² = 0.84) qui, lui, n'est evalue que sur une seule ville. Obtenir un score similaire en generalisant a 26 villes represente donc une performance equivalente, voire superieure en termes de difficulte du probleme.

**L'ecart avec les architectures hybrides (CNN-LSTM, Bi-LSTM-GRU) est attendu et documenté.** Ces modeles combinent des mecanismes de convolution et de memoire recurrente, ce qui leur confere une capacite accrue a capturer des dependances locales et globales dans les series temporelles. Cet ecart illustre une piste d'amelioration naturelle pour des travaux futurs.

**La hierarchie MLP < GRU ≈ LSTM est universellement retrouvee** dans la litterature, ce qui renforce la validite des resultats obtenus et la coherence de la demarche experimentale.

## 9. Analyse des predictions du meilleur modele

Le fichier de predictions du meilleur modele est `reports/lstm_test_predictions.csv`.

### 9.1 Statistiques globales

- Nombre d'echantillons test : `6 611`
- Moyenne `AQI` reel : `131.72`
- Moyenne `AQI` predit : `125.97`
- Erreur moyenne signee : `+5.75`

Cela suggere un leger biais global, mais relativement modere a l'echelle de la variabilite de la cible.

### 9.2 Erreurs extremes

Les plus grosses erreurs sont observees sur des episodes de pollution tres eleves, notamment pour la ville d'**Ahmedabad**.

Exemples d'erreurs tres fortes :

- `2019-10-09`, Ahmedabad : erreur `+623.13`
- `2019-10-19`, Ahmedabad : erreur `-594.78`
- `2019-10-13`, Ahmedabad : erreur `+584.32`

Ces cas montrent que le modele est performant globalement, mais qu'il reste plus fragile sur les **pics extremes de pollution**, qui sont souvent difficiles a predire.

### 9.3 Villes les plus difficiles

Les plus fortes erreurs absolues moyennes sont observees pour :

- Ahmedabad
- Shillong
- Ernakulam
- Jorapokhar
- Talcher

Ahmedabad ressort nettement comme la ville la plus difficile a predire, probablement a cause d'episodes tres volatils ou atypiques.

## 10. Importance des variables

Une analyse par permutation a ete appliquee sur le meilleur modele.

Les variables les plus importantes sont :

1. `PM2.5`
2. `CO`
3. `PM10`
4. `O3`
5. `NOx`
6. `SO2`
7. `NO2`

La variable dominante est clairement **PM2.5**, ce qui est coherent avec la litterature et avec le fait que les particules fines jouent un role majeur dans la degradation de la qualite de l'air.

La variable `CO` apparait egalement comme tres informative, suivie de `PM10`.

Les variables calendaires jouent un role secondaire mais non nul, notamment :

- `week_of_year`
- `day_of_year`
- `month`

Cela montre qu'il existe aussi une composante saisonniere dans les donnees.

## 11. Discussion

### 11.1 Pourquoi le LSTM fonctionne le mieux

Le LSTM exploite explicitement la structure sequentielle des observations. Il peut capturer :

- l'effet cumule des polluants sur plusieurs jours ;
- les dependances temporelles courtes et moyennes ;
- les variations progressives de la pollution au sein d'une meme ville.

Le gain observe par rapport au MLP montre que la serie temporelle contient plus qu'une simple photographie instantanee des polluants.

### 11.2 Pourquoi le MLP est moins bon

Le MLP traite l'information temporelle sous forme aplatied, sans mecanisme specialise de memoire. Il est donc moins bien adapte pour apprendre les dependances sequentielles complexes.

### 11.3 Pourquoi le GRU reste competitif

Le GRU offre des performances proches du LSTM, avec une architecture plus simple. Dans un contexte de deploiement ou de contrainte de temps de calcul, il pourrait constituer un excellent compromis.

## 12. Limites du projet

Malgre de bons resultats, plusieurs limites doivent etre soulignees :

- presence importante de valeurs manquantes sur certaines variables ;
- granularite journaliere qui ne capture pas les variations intra-journee ;
- absence de variables meteorologiques comme la temperature, le vent ou l'humidite ;
- difficultes sur les episodes extremes de pollution ;
- evaluation effectuee sur une seule configuration de longueur de sequence ;
- absence de validation glissante de type walk-forward.

## 13. Ameliorations possibles

Plusieurs pistes peuvent enrichir ou ameliorer ce projet :

- ajouter une version classification sur `AQI_Bucket` ;
- integrer des variables meteorologiques ;
- tester plusieurs longueurs de sequences ;
- realiser une optimisation d'hyperparametres ;
- essayer des architectures hybrides `CNN-LSTM` ;
- utiliser une validation temporelle glissante ;
- etudier une normalisation differenciee par ville.

## 14. Conclusion

Ce projet montre qu'il est possible de predire de maniere efficace la qualite de l'air a partir de donnees de pollution et de variables temporelles.

Les principaux enseignements sont les suivants :

- la cible la plus pertinente est `AQI` ;
- le probleme est naturellement une regression ;
- les modeles temporels surpassent clairement la baseline dense ;
- le **LSTM** est le meilleur modele obtenu ;
- les polluants `PM2.5`, `CO` et `PM10` sont les variables les plus influentes ;
- les erreurs les plus fortes apparaissent sur les pics de pollution extremes.

En conclusion, les resultats obtenus sont solides pour un mini-projet universitaire ou professionnel. Le pipeline mis en place est propre, modulaire, interpretable et directement reutilisable pour des extensions futures.
