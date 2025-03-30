import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import lime
import lime.lime_text
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("phishing_email_dataset.csv")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, random_state=42)

# Create pipeline
vectorizer = TfidfVectorizer()
model = RandomForestClassifier(n_estimators=100, random_state=42)
pipeline = make_pipeline(vectorizer, model)
pipeline.fit(X_train, y_train)

# Explain a sample
explainer = lime.lime_text.LimeTextExplainer(class_names=["legitimate", "phishing"])
idx = 1
exp = explainer.explain_instance(X_test.iloc[idx], pipeline.predict_proba, num_features=6)

# Save the explanation
fig = exp.as_pyplot_figure()
fig.savefig("lime_explanation_phishguard.png", bbox_inches="tight")
