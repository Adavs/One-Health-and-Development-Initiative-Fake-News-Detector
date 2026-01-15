import streamlit as st
import pickle

# Load the trained AI
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# App Styling for OHDI
st.title("🛡️ OHDI: One Health Misinformation Guard")
st.markdown("Verifying news at the intersection of Human, Animal, and Environmental Health.")

# User Input
user_input = st.text_area("Paste the health news article or social media post here:")

if st.button("Analyze Veracity"):
    if user_input:
        # Transform input and predict
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)
        
        # Display Result
        if prediction[0] == 'REAL':
            st.success("✅ This article appears to be RELIABLE based on linguistic patterns.")
        else:
            st.error("🚨 WARNING: This content shows signs of misinformation/fake news.")
    else:
        st.write("Please enter some text to analyze.")