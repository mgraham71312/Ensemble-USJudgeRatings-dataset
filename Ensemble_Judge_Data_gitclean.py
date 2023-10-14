# libraries
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold 
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


# Step 1: Import
df = pd.read_csv("USJudgeRatings.csv")

# Dichotomize RTEN; random/arbitrary level (8.0) selected
# Less than 8.0 will be deemed "unworthy" (0) 
# 8.0 or greater will be "worthy" (1)
df.RTEN[df.RTEN < 8.0] = 0
df.RTEN[df.RTEN >= 8.0] = 1

X = df[['CONT', 'INTG', 'DMNR', 'DILG', 'CFMG', 'DECI', 
        'PREP', 'FAMI', 'ORAL', 'WRIT', 'PHYS']].values
y = df[['RTEN']].values.ravel()
indep = ['Contact', 'Integrity', 'Demeanor', 'Diligence', 
         'Case Flow', 'Decision', 'Preparation', 'Familiarity', 
         'Oral', 'Written', 'Physical']
fold = KFold(n_splits=3, shuffle=True, random_state=42)


# Step 2: Decision tree.
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)
dt_accuracy = cross_val_score(dt, X, y, scoring='accuracy', cv=fold).mean()
print("Decision tree accuracy:", dt_accuracy)  

plt.figure(figsize=(7,7))
tree.plot_tree(dt, feature_names=indep, 
               class_names=["Not Worthy", "Worthy"], 
               fontsize=10)


# Step 3: Bagging
bagging = DecisionTreeClassifier(random_state=42)     

bag10 = BaggingClassifier(estimator=bagging, n_estimators=10, random_state=42)
bag10_accuracy = cross_val_score(bag10, X, y, scoring='accuracy', cv=fold).mean()
print("Bagging with 10 trees accuracy: ", bag10_accuracy)

bag100 = BaggingClassifier(estimator=bagging, n_estimators=100, random_state=42)
bag100_accuracy = cross_val_score(bag100, X, y, scoring='accuracy', cv=fold).mean()
print("Bagging with 100 trees accuracy: ", bag100_accuracy)

bag500 = BaggingClassifier(estimator=bagging, n_estimators=500, random_state=42)
bag500_accuracy = cross_val_score(bag500, X, y, scoring='accuracy', cv=fold).mean()
print("Bagging with 500 trees accuracy: ", bag500_accuracy)


# Step 4: Random Forest
rf10 = RandomForestClassifier(n_estimators=10, random_state=42)                  
rf10_accuracy = cross_val_score(rf10, X, y, scoring='accuracy', cv=fold).mean()
print("Random Forest with 10 trees accuracy: ", rf10_accuracy)

rf100 = RandomForestClassifier(n_estimators=100, random_state=42)
rf100_accuracy = cross_val_score(rf100, X, y, scoring='accuracy', cv=fold).mean()
print("Random Forest with 100 trees accuracy: ", rf100_accuracy)

rf500 = RandomForestClassifier(n_estimators=500, random_state=42)
rf500_accuracy = cross_val_score(rf500, X, y, scoring='accuracy', cv=fold).mean()
print("Random Forest with 500 trees accuracy: ", rf500_accuracy)


# References:
# New Haven Register. (1977, January 14). Lawyers' ratings of state judges in the US Superior Court. Retrieved April 9, 2022, from https://vincentarelbundock.github.io/Rdatasets/datasets.html
