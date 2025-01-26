
# Credit Card Fraud Detection
This project analyzes a credit card fraud dataset using Python

# Steps

# Step 1: Import Libraries

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
```
# Step 2: Load the Dataset
```
data = pd.read_csv('Creditcard_data.csv')  
```
# Step 3: Initial Exploration of the Dataset
```
print(data.head())
print(data.info())
```
data.head(): To display the first 5 rows of the dataset.

data.info(): Provides metadata like column names, data types, and non-null counts.

# Step 4: Defining x and y
```
X = data.drop("Class", axis=1)
y = data["Class"]
```

# Step 5: Handle Imbalanced Dataset with SMOTE
```
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```
SMOTE (Synthetic Minority Oversampling Technique): Balances the dataset by creating synthetic samples for the minority class.

x: Independent features.

y: Target variable.

fit_resample: Applies SMOTE to create x_smote and y_smote, balanced datasets.

# Step 6: Checking the new balanced data
'''
print("Original class distribution:")
print(y.value_counts())
print("Resampled class distribution:")
print(y_resampled.value_counts())
'''

# Step 7: Create Samples

 ```
sample_size = 1000
samples = {
    "Sampling1": X_resampled.sample(n=sample_size, random_state=42),
    "Sampling2": X_resampled.sample(n=sample_size, random_state=21),
    "Sampling3": X_resampled.iloc[::len(X_resampled)//sample_size, :],
    "Sampling4": X_resampled.sample(n=sample_size, random_state=56),
    "Sampling5": X_resampled.sample(n=sample_size, random_state=99),
}

sample_datasets = {
    name: (sample, y_resampled.loc[sample.index])
    for name, sample in samples.items()
}

```
Multiple sampling techniques are applied to the balanced dataset to extract samples:
1. Simple Random Sampling: Selects a random subset of the data without replacement.
2. Stratified Sampling: Ensures that the class distribution is maintained in the sample.
3. Systematic Sampling: Selects samples based on a fixed interval k.
4. Cluster Sampling: Divides the dataset into clusters and selects one cluster.
5. Bootstrapping: Samples data points with replacement.

# Step 8: Defining the Models to be used 
```
models = {
    "M1": LogisticRegression(random_state=42),
    "M2": DecisionTreeClassifier(random_state=42),
    "M3": RandomForestClassifier(random_state=42),
    "M4": SVC(random_state=42),
    "M5": KNeighborsClassifier(),
}

results = {}
```
# Step 9: Creating a result matrix by training and evaluating various models
```
for sample_name, (X_sample, y_sample) in sample_datasets.items():
    print(f"Evaluating for {sample_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        if sample_name not in results:
            results[sample_name] = {}
        results[sample_name][model_name] = accuracy

matrix_data = []
for sample_name, accuracies in results.items():
    row = [accuracies.get(model, None) for model in models.keys()]
    matrix_data.append(row)

results_matrix = pd.DataFrame(
    matrix_data,
    index=results.keys(),
    columns=models.keys()
)

print("Accuracy Matrix (Assignment Format):")
print(results_matrix)
```
# Step 10: Train and evaluate various models
```

from sklearn.metrics import accuracy_score

for model_name, model in models.items():
    results[model_name] = []
    for i, sample in enumerate(samples):

        X_sample = sample.drop('Class', axis=1)
        y_sample = sample['Class']

        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name].append(accuracy)


results_df = pd.DataFrame(results, index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"])
print(results_df)

results_df.to_csv("model_accuracies.csv")
```
# Step 11: Storing the data in .csv file and finding the best combinations
```
best_combinations = results_matrix.idxmax()
print("Best Sampling Technique for Each Model:")
print(best_combinations)

results_matrix.to_csv("results_matrix_assignment_exact.csv")
```
