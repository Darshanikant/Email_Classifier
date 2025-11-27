import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load(r"C:\Users\sunil\Desktop\NIT Intership proj\Project List\3. Email_Spam_Detection\spam_classifier_model.pkl")
vectorizer = joblib.load(r"C:\Users\sunil\Desktop\NIT Intership proj\Project List\3. Email_Spam_Detection\tfidf_vectorizer.pkl")

# Streamlit UI
st.title("📧 Email Spam Detector")
st.markdown("Enter the content of an email to check if it's **Spam** or **Not Spam**.")

st.divider()

# Input area
email_text = st.text_area("Paste your email message below:", height=200)

# Predict button
if st.button("Detect Spam"):
    if not email_text.strip():
        st.warning("⚠️ Please enter an email to analyze.")
    else:
        # Vectorize and predict
        email_tfidf = vectorizer.transform([email_text])
        prediction = model.predict(email_tfidf)[0]

        # Display result
        st.subheader("Prediction:")
        if prediction == 1:
            st.error("🚫 Spam Email")
        else:
            st.success("✅ Not Spam Email")

# Footer
st.divider()
st.markdown("Developed by **Darshanikanta**")
