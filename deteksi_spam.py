
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Ubah label ke angka
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Vektorisasi teks
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

# Evaluasi
print("Naive Bayes Accuracy:", accuracy_score(y_test, pred_nb))
print("Decision Tree Accuracy:", accuracy_score(y_test, pred_dt))
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, pred_nb))

# Visualisasi distribusi label
df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribusi Label Spam vs Ham')
plt.xlabel('Label')
plt.ylabel('Jumlah Pesan')
plt.tight_layout()
plt.show()
