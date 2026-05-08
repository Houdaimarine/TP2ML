# TP2ML
#Classification du Cancer du Sein : Comparaison de Modèles ML

-Présentation du Projet
Ce projet implémente un pipeline complet de Machine Learning pour la classification binaire (Maligne / Bénigne) en utilisant le célèbre jeu de données Breast Cancer Wisconsin. L'objectif principal est de comparer les performances de plusieurs algorithmes et de valider leur robustesse via une validation croisée stratifiée.

-Étapes de Réalisation
Le script `tp.py` suit une démarche scientifique rigoureuse :
1.  Exploration des données : Analyse statistique descriptive et visualisation de la distribution des classes.
2.  Prétraitement : Normalisation des données avec `StandardScaler` pour garantir que toutes les caractéristiques contribuent équitablement au modèle.
3.  Entraînement Multimodèle : Évaluation comparative de 5 algorithmes :
    *   Régression Logistique
    *   K-Nearest Neighbors (KNN)
    *   Arbres de Décision
    *   Random Forest (Modèle principal)
    *   Support Vector Machine (SVM)
4.  Validation Robuste : Utilisation de la `StratifiedKFold` pour assurer une évaluation stable sur différentes partitions des données.
5.  Diagnostic: Génération de courbes d'apprentissage (`Learning Curves`) et matrice de confusion.

-Visualisations Incluses
Le projet génère automatiquement plusieurs graphiques pour l'analyse :
*   Histogrammes : Distribution de chaque caractéristique physique.
*   Matrice de Corrélation : Identification des relations entre les variables.
*   Barplots de Comparaison : Comparaison visuelle de l'accuracy de chaque modèle.
*   Matrice de Confusion : Détails des prédictions (Vrais Positifs, Faux Négatifs, etc.) pour le Random Forest.
*   Learning Curve : Analyse du compromis Biais/Variance.

-Installation et Prérequis
Assurez-vous d'avoir Python installé ainsi que les bibliothèques suivantes :
pip install pandas numpy matplotlib seaborn scikit-learn

Développé par : Houda Imarine

-Contexte Académique
Ce travail a été réalisé dans le cadre de ma formation en Master Intelligence Artificielle. Il illustre la mise en pratique des concepts théoriques de classification supervisée et d'évaluation de la performance des modèles.
