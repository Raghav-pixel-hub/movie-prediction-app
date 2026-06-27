import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style('whitegrid')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎬 Movie Score Classifier", layout="wide")

st.title("🎬 Movie IMDB Score Classifier")
st.markdown(
    "This app walks through every step of a **beginner machine learning project**: "
    "from loading data and exploring it, all the way to training models and evaluating them."
)
st.info("📂 **First step:** upload your `movie_metadata.csv` file in the sidebar to get started.")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload movie_metadata.csv", type=["csv"])
    st.markdown("---")
    st.markdown(
        "**Steps in this app:**\n"
        "1. Load & Preview Data\n"
        "2. Explore the Data (EDA)\n"
        "3. Clean & Prepare Data\n"
        "4. Feature Engineering\n"
        "5. Train & Compare Models\n"
        "6. Evaluate Best Model\n"
    )

if uploaded_file is None:
    st.stop()

# ─── STEP 1: Load Data ────────────────────────────────────────────────────────
st.markdown("---")
st.header("Step 1 – Load & Preview the Data")
st.markdown(
    "We start by loading the CSV file into a **pandas DataFrame** — "
    "a table structure that makes it easy to work with data in Python."
)

df_raw = pd.read_csv(uploaded_file)

col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", df_raw.shape[0])
col2.metric("Total Columns", df_raw.shape[1])
col3.metric("Duplicate Rows", df_raw.duplicated().sum())

st.subheader("First 5 rows")
st.dataframe(df_raw.head(), use_container_width=True)

with st.expander("📋 Column data types"):
    dtypes_df = pd.DataFrame({"Column": df_raw.dtypes.index, "Type": df_raw.dtypes.values})
    st.dataframe(dtypes_df, use_container_width=True)

# ─── STEP 2: Exploratory Data Analysis ───────────────────────────────────────
st.markdown("---")
st.header("Step 2 – Explore the Data (EDA)")
st.markdown(
    "Before building any model we need to **understand** our data: "
    "what values are common, what's missing, and how features relate to each other."
)

# 2a – Missing values
st.subheader("2a · Missing Values")
missing = df_raw.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    st.success("No missing values found!")
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing.values, y=missing.index, color='steelblue', ax=ax)
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Column")
    ax.set_title("Missing Values per Column")
    st.pyplot(fig)
    plt.close()

# 2b – IMDB Score distribution
st.subheader("2b · Distribution of IMDB Score (Target Variable)")
st.markdown(
    "The **IMDB score** is what we want to predict. "
    "Let's see how it's spread across the dataset."
)

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df_raw['imdb_score'], bins=30, kde=True, color='darkorange', ax=ax)
ax.axvline(df_raw['imdb_score'].mean(), color='red', linestyle='--',
           label=f"Mean = {df_raw['imdb_score'].mean():.2f}")
ax.set_xlabel("IMDB Score")
ax.set_ylabel("Number of Movies")
ax.set_title("Distribution of IMDB Scores")
ax.legend()
st.pyplot(fig)
plt.close()

# 2c – Numeric feature distributions
st.subheader("2c · Numeric Feature Distributions")
num_features = ['duration', 'budget', 'gross', 'num_voted_users',
                'num_critic_for_reviews', 'title_year']
num_features = [f for f in num_features if f in df_raw.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(num_features):
    sns.histplot(df_raw[col].dropna(), bins=30, kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 2d – Scatter plots vs IMDB Score
st.subheader("2d · Features vs IMDB Score")
scatter_features = ['budget', 'duration', 'num_voted_users', 'num_critic_for_reviews']
scatter_features = [f for f in scatter_features if f in df_raw.columns]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
for i, col in enumerate(scatter_features):
    sns.scatterplot(x=df_raw[col], y=df_raw['imdb_score'], ax=axes[i], alpha=0.4, color='purple')
    axes[i].set_title(f"{col} vs IMDB Score")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 2e – Categorical breakdowns
st.subheader("2e · Categorical Feature Breakdown")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

if 'content_rating' in df_raw.columns:
    cr = df_raw['content_rating'].value_counts().head(8)
    sns.barplot(x=cr.values, y=cr.index, ax=axes[0], color='coral')
    axes[0].set_title("Top Content Ratings")
    axes[0].set_xlabel("Count")

if 'country' in df_raw.columns:
    cc = df_raw['country'].value_counts().head(8)
    sns.barplot(x=cc.values, y=cc.index, ax=axes[1], color='seagreen')
    axes[1].set_title("Top 8 Countries")
    axes[1].set_xlabel("Count")

if 'content_rating' in df_raw.columns:
    avg = df_raw.groupby('content_rating')['imdb_score'].mean().sort_values(ascending=False).head(8)
    sns.barplot(x=avg.values, y=avg.index, ax=axes[2], color='slateblue')
    axes[2].set_title("Avg IMDB Score by Content Rating")
    axes[2].set_xlabel("Average IMDB Score")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# 2f – Correlation heatmap (raw)
st.subheader("2f · Correlation Heatmap (Raw Data)")
st.markdown(
    "A **correlation heatmap** shows how numeric features relate to each other. "
    "Values close to 1 or -1 mean a strong relationship."
)
fig, ax = plt.subplots(figsize=(12, 9))
numeric_df = df_raw.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', center=0, ax=ax)
ax.set_title("Correlation Heatmap of Numeric Features (raw data)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── STEP 3: Clean & Prepare Data ────────────────────────────────────────────
st.markdown("---")
st.header("Step 3 – Clean & Prepare the Data")
st.markdown(
    "Raw data is messy. We need to: remove duplicates, drop useless columns, "
    "fill in missing values, and convert text columns to numbers."
)

df = df_raw.copy()

# 3a – Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
st.write(f"✅ Removed **{before - after}** duplicate rows ({before} → {after})")

# 3b – Drop identifier/free-text columns
columns_to_drop = ['movie_imdb_link', 'plot_keywords', 'movie_title',
                   'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
columns_to_drop = [c for c in columns_to_drop if c in df.columns]
df = df.drop(columns=columns_to_drop)
st.write(f"✅ Dropped identifier/free-text columns: `{columns_to_drop}`")

# 3c – Extract main genre
if 'genres' in df.columns:
    df['main_genre'] = df['genres'].apply(lambda x: str(x).split('|')[0])
    df = df.drop(columns=['genres'])
    st.write("✅ Extracted `main_genre` from `genres` column (kept first genre only)")

# 3d – Fill missing values
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

st.write(f"✅ Filled **{len(numeric_cols)}** numeric columns with **median**")
st.write(f"✅ Filled **{len(categorical_cols)}** categorical columns with **mode** (most common value)")
st.success(f"Total missing values remaining: **{df.isnull().sum().sum()}**")

# 3e – Label encoding
st.subheader("3a · Label Encoding Categorical Columns")
st.markdown(
    "Machine learning models only understand numbers. "
    "**Label Encoding** converts each text category to an integer."
)
label_encoders = {}
encoding_summary = []
for col in categorical_cols:
    if col in df.columns:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        label_encoders[col] = enc
        encoding_summary.append({"Column": col, "Unique Categories": len(enc.classes_)})

st.dataframe(pd.DataFrame(encoding_summary), use_container_width=True)

# ─── STEP 4: Feature Engineering ─────────────────────────────────────────────
st.markdown("---")
st.header("Step 4 – Feature Engineering")

# 4a – Multicollinearity check
st.subheader("4a · Correlation Heatmap (After Cleaning)")
feature_cols_for_corr = [c for c in df.columns if c != 'imdb_score']
fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(df[feature_cols_for_corr].corr(), annot=False, cmap='coolwarm', center=0, ax=ax)
ax.set_title("Correlation Heatmap After Cleaning & Encoding")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 4b – Drop high multicollinearity column
if 'cast_total_facebook_likes' in df.columns:
    df = df.drop(columns=['cast_total_facebook_likes'])
    st.write("✅ Dropped `cast_total_facebook_likes` (very high multicollinearity with actor like columns)")

# 4c – Create target classification column
st.subheader("4b · Create Target Variable: Classify")
st.markdown(
    "We convert the numeric IMDB score into **3 classes** for classification:\n"
    "- 🔴 **Flop** → score 1–3\n"
    "- 🟡 **Average** → score 3–6\n"
    "- 🟢 **Hit** → score 6–10"
)

def classify_score(score):
    if score <= 3:
        return 'Flop'
    elif score <= 6:
        return 'Average'
    else:
        return 'Hit'

df['Classify'] = df['imdb_score'].apply(classify_score)
class_counts = df['Classify'].value_counts()

col1, col2, col3 = st.columns(3)
col1.metric("🔴 Flop", class_counts.get("Flop", 0))
col2.metric("🟡 Average", class_counts.get("Average", 0))
col3.metric("🟢 Hit", class_counts.get("Hit", 0))

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='Classify', data=df, order=['Flop', 'Average', 'Hit'], palette='viridis', ax=ax)
ax.set_title("Movie Classification Based on IMDB Score")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Movies")
st.pyplot(fig)
plt.close()

# 4d – Encode target & scale features
target_encoder = LabelEncoder()
df['Classify_encoded'] = target_encoder.fit_transform(df['Classify'])
class_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
st.write("Class label mapping:", class_mapping)

feature_columns = [c for c in df.columns if c not in ['imdb_score', 'Classify', 'Classify_encoded']]
X = df[feature_columns]
y = df['Classify_encoded']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

st.subheader("4c · Feature Scaling (StandardScaler)")
st.markdown(
    "Features have very different scales (budget in millions, duration in minutes). "
    "**StandardScaler** makes every feature have mean=0 and std=1 so no single feature dominates."
)

col1, col2 = st.columns(2)
with col1:
    st.write("**Before scaling (sample)**")
    st.dataframe(X.head(3), use_container_width=True)
with col2:
    st.write("**After scaling (sample)**")
    st.dataframe(X_scaled.head(3).round(3), use_container_width=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
st.info(f"📊 Data split → **Training set:** {X_train.shape[0]} rows | **Test set:** {X_test.shape[0]} rows (80/20 split)")

# ─── STEP 5: Train & Compare Models ──────────────────────────────────────────
st.markdown("---")
st.header("Step 5 – Train & Compare Models")
st.markdown(
    "We train **4 different classifiers** and compare their accuracy on the test set. "
    "The best model will be evaluated in detail."
)

with st.spinner("Training models… this may take a moment ⏳"):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    }
    accuracy_results = {}
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracy_results[model_name] = acc
        trained_models[model_name] = model

best_model_name = max(accuracy_results, key=accuracy_results.get)

# Display metrics
cols = st.columns(4)
for i, (name, acc) in enumerate(accuracy_results.items()):
    icon = "🏆" if name == best_model_name else ""
    cols[i].metric(f"{icon} {name}", f"{acc:.2%}")

# Bar chart
fig, ax = plt.subplots(figsize=(8, 4))
bars = sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()),
                   color='mediumseagreen', ax=ax)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison – Accuracy on Test Set")
ax.set_ylim(0, 1)
for i, v in enumerate(accuracy_results.values()):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.success(f"🏆 Best model: **{best_model_name}** with **{accuracy_results[best_model_name]:.2%}** accuracy")

# ─── STEP 6: Evaluate Best Model (Random Forest) ─────────────────────────────
st.markdown("---")
st.header("Step 6 – Evaluate the Best Model (Random Forest)")
st.markdown(
    "Now we take a deep look at the **Random Forest** model — "
    "the star of this project. We'll look at its confusion matrix, "
    "classification report, and which features it found most important."
)

rf_model = trained_models['Random Forest']
rf_preds = rf_model.predict(X_test)
class_names = target_encoder.classes_

# 6a – Confusion Matrix
st.subheader("6a · Confusion Matrix")
st.markdown(
    "A **confusion matrix** shows exactly which classes the model got right "
    "and which ones it mixed up. Diagonal = correct predictions."
)
cm = confusion_matrix(y_test, rf_preds)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Random Forest – Confusion Matrix")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 6b – Classification Report
st.subheader("6b · Classification Report")
st.markdown(
    "**Precision** = of all the movies predicted as X, how many really were X?\n\n"
    "**Recall** = of all the actual X movies, how many did the model find?\n\n"
    "**F1-score** = the balance between precision and recall."
)
report_dict = classification_report(y_test, rf_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).T
st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

# 6c – Feature Importance
st.subheader("6c · Top 12 Most Important Features")
st.markdown(
    "Random Forest tells us which features it relied on the most "
    "when making predictions — very useful for understanding the data!"
)
importances = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=importances.values[:12], y=importances.index[:12], color='darkcyan', ax=ax)
ax.set_xlabel("Importance Score")
ax.set_title("Top 12 Most Important Features (Random Forest)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.write("**Top 5 features:**")
st.dataframe(importances.head(5).reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
             use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "✅ **All steps complete!** You've gone from raw CSV data all the way to a trained "
    "and evaluated Random Forest classifier. Great work! 🎉"
)