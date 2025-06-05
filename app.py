
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Preprocessing
df = df.drop("User ID", axis=1)  # Rimuoviamo l'ID che non serve
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])  # Male = 1, Female = 0

# Separazione X e y
X = df[["Gender", "Age", "EstimatedSalary"]]
y = df["Purchased"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modello SVM
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Streamlit UI
st.title("📊 Predict Ad Clicks with SVM")
st.write("Inserisci i dati utente per prevedere se cliccherà su un annuncio.")

# Input utente
gender = st.selectbox("Genere", ["Maschio", "Femmina"])
age = st.slider("Età", 18, 60, 25)
salary = st.slider("Salario stimato (€)", 15000, 150000, 50000, step=1000)

# Conversione input
gender_val = 1 if gender == "Maschio" else 0
input_data = pd.DataFrame([[gender_val, age, salary]], columns=["Gender", "Age", "EstimatedSalary"])

# Predizione
if st.button("Prevedi"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ L'utente PROBABILMENTE cliccherà sull'annuncio.")
    else:
        st.warning("❌ L'utente PROBABILMENTE NON cliccherà sull'annuncio.")

# Valutazione modello
st.subheader("📈 Performance del modello")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Click", "Click"], yticklabels=["No Click", "Click"])
plt.xlabel("Predetto")
plt.ylabel("Reale")
st.pyplot(fig)

st.text("Report classificazione:")
st.text(classification_report(y_test, y_pred))
