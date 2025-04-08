import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Load the dataset
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

df.count()

# Preview the data
print("First 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())


# Drop all rows with missing values

df = df.dropna()
# Final structure of the cleaned dataset
print("\nCleaned dataset info:")
df.info()


df.columns

#Data leakage considerations

df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

#Data Granularity

#Location selection
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()
#Extracting a seasonality feature

#Create a function to map dates to seasons

def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'
    

# Task 1: Map the dates to seasons and drop the Date column
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

# Drop the original 'Date' column
df = df.drop(columns=['Date'])

# Display the updated DataFrame
df
print(df['Season'].value_counts())

# Task 2. Define the feature and target dataframes

print(df.columns)
X = df.drop(columns='Rainfall', axis=1)
y = df['Rainfall']

# Task  3. How balanced are the classes?

y.value_counts()
#y.value_counts(normalize=True)


#Task 4. What can you conclude from these counts?
print(y.value_counts())
print("\nClass percentages:")
print(y.value_counts(normalize=True)*100)

# Task  5. Split data into training and test sets, ensuring target stratification

import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)

# Drop missing values
df_clean = df.dropna()

# Define X and y from the cleaned DataFrame
X = df_clean.drop('RainTomorrow', axis=1)
y = df_clean['RainTomorrow']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
# Task 6. Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features


# Detect numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Define separate transformers for both feature types and combine them into a single preprocessing transformer

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Task 7. Combine the transformers into a single preprocessing column transformer


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

#Task  8. Create a pipeline by combining the preprocessing with a Random Forest classifier
from tempfile import mkdtemp

# Specify a temporary directory for caching
cachedir = mkdtemp()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
], memory=cachedir)
pipeline.fit(X_train, y_train)

# Define a parameter grid to use in a cross validation grid search model optimizer

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}


# Perform grid search cross-validation and fit the best model to the training data

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameter grid for RandomForestClassifier
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

# Setup GridSearchCV with the pipeline
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',  # or 'f1', 'roc_auc' depending on your objective
    n_jobs=-1,
    verbose=1
)

# Fit grid search on training data
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Optional: Check best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# task  9. Instantiate and fit GridSearchCV to the pipeline

grid_search = GridSearchCV(
    pipeline,                  # your pipeline with preprocessing + classifier
    param_grid,                # the hyperparameter grid
    cv=cv,                     # StratifiedKFold or any other CV strategy
    scoring='accuracy',        # or 'f1', 'roc_auc', etc.
    verbose=2,
    n_jobs=-1
)

# Fit to the training data
grid_search.fit(X_train, y_train)


print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Task 10. Display your model's estimated score
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# Task 11. Get the model predictions from the grid search estimator on the unseen data
y_pred = grid_search.predict(X_test)

# task 12. Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Task 13. Plot the confusion matrix
 
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Task 14. Extract the feature importances
# Extract feature importances from the classifier inside the pipeline
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_


# Combine numeric and categorical feature names
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

N = 20  # Change this number to display more or fewer features
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

top_features = importance_df.head(N)

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()

# Task 15. Update the pipeline and the parameter grid

# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# Update the model's estimator to use the new pipeline
grid_search.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

# Update the parameter grid in GridSearchCV
grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
model = grid_search.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#Compare the results to your previous model.
print(classification_report(y_test, y_pred))
# Generate the confusion matrix 

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()
