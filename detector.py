import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# 1. Merge all CSV files in the 'data' folder
path = 'data'
all_files = glob.glob(os.path.join(path, "*.csv"))
li = []

for filename in all_files:
    # Read the file and strip any accidental whitespace from headers
    temp_df = pd.read_csv(filename)
    temp_df.columns = temp_df.columns.str.strip().str.lower()
    cols = temp_df.columns.tolist()
    
    # --- SMART COLUMN MAPPING ---
    # Find the 'text' column
    text_col = next((c for c in cols if c in ['text', 'body', 'article', 'question']), None)
    if text_col:
        temp_df.rename(columns={text_col: 'text'}, inplace=True)
    
    # Find or create the 'label' column
    label_col = next((c for c in cols if c in ['label', 'class', 'target', 'veracity']), None)
    if label_col:
        temp_df.rename(columns={label_col: 'label'}, inplace=True)
    elif 'true' in filename.lower():
        temp_df['label'] = 'REAL'
    elif 'fake' in filename.lower():
        temp_df['label'] = 'FAKE'
    
    # Only keep the standardized columns to avoid errors during merge
    if 'text' in temp_df.columns and 'label' in temp_df.columns:
        li.append(temp_df[['text', 'label']])

# Combine everything into one master dataframe
df = pd.concat(li, axis=0, ignore_index=True)
df.dropna(inplace=True)

# 2. Train the AI Model
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 3. Save the results
if not os.path.exists('models'):
    os.makedirs('models')

pickle.dump(pac, open('models/model.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('models/vectorizer.pkl', 'wb'))

print(f"✅ Success! Trained on {len(df)} rows across {len(all_files)} files.")