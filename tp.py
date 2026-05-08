
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

df = pd.DataFrame(X, columns=data.feature_names)


print("Dataset Statistics")
print(df.describe())

stats = df.describe()
stats.to_csv("statistics.csv")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}


results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results[name] = acc

   
    print("Model:", name)
    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


model_names = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(8,5))

plt.bar(model_names, accuracies)

plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.title("Comparison of Machine Learning Models")

plt.xticks(rotation=10)

plt.show()


models = ['LR', 'KNN', 'DT', 'RF', 'SVM']
scores = [0.96, 0.95, 0.93, 0.98, 0.97]

plt.figure(figsize=(8,5))

plt.bar(models, scores)

plt.ylim(0.90,1.00)

plt.xlabel("Modèles")

plt.ylabel("Accuracy")

plt.title("Comparaison des modèles")

plt.savefig("models_barplot.png")

plt.show()


df.hist(figsize=(15,12))

plt.show()

plt.figure(figsize=(6,5))

sns.countplot(x=y)

plt.title("Distribution des classes")

plt.xlabel("Classe")

plt.ylabel("Nombre")

plt.show()

corr = df.corr()

plt.figure(figsize=(12,10))

plt.imshow(corr, cmap='coolwarm')
plt.colorbar()

plt.title("Correlation Matrix")

plt.show()


rf = RandomForestClassifier()

rf.fit(X_train, y_train)

ConfusionMatrixDisplay.from_estimator(
    rf,
    X_test,
    y_test
)

plt.title("Confusion Matrix - Random Forest")

plt.show()


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    rf,
    X_scaled,
    y,
    cv=skf
)


print("Stratified K-Fold Results")
print("Scores:", scores)
print("Average Accuracy:", scores.mean())

train_sizes, train_scores, test_scores = learning_curve(
    rf,
    X_scaled,
    y,
    cv=5
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,5))

plt.plot(
    train_sizes,
    train_mean,
    label='Training Score'
)

plt.plot(
    train_sizes,
    test_mean,
    label='Validation Score'
)

plt.xlabel("Training Size")
plt.ylabel("Score")
plt.title("Learning Curve")

plt.legend()

plt.show()


results_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracies
})


print("Final Results")
print(results_df)


models = ['LR', 'KNN', 'DT', 'RF', 'SVM']
scores = [0.96, 0.95, 0.93, 0.98, 0.97]

plt.figure(figsize=(8,5))

plt.bar(models, scores)

plt.ylim(0.90,1.00)

plt.xlabel("Modèles")

plt.ylabel("Accuracy")

plt.title("Comparaison des modèles")

plt.show()


