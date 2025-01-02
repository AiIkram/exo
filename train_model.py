import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('Financial_inclusion_dataset.csv')

# Drop unnecessary columns
X = df.drop(columns=['year', 'uniqueid', 'bank_account'], errors='ignore')

# Encode target column
target_column = "bank_account"
le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column])
y = df[target_column]

# Define categorical columns
categorical_columns = [
    "country", "location_type", "cellphone_access", "gender_of_respondent",
    "relationship_with_head", "marital_status", "education_level", "job_type"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ],
    remainder="passthrough"
)

# Define the pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Define GridSearchCV parameters
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
}
grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring="accuracy", verbose=3, n_jobs=-1
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
grid_search.fit(X_train, y_train)

# Evaluate the model
y_pred = grid_search.best_estimator_.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(grid_search.best_estimator_, "rf_model.pkl")
print("Model saved as rf_model.pkl")
