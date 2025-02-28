# 📩 Spam Detection Model Using Multinomial Naive Bayes

## 📌 Project Description
This project focuses on building a **Spam Detection Model** using **Multinomial Naive Bayes**. The dataset consists of labeled messages categorized as either **'spam'** (unwanted messages) or **'ham'** (legitimate messages). The goal is to classify incoming text messages accurately using **Natural Language Processing (NLP)** techniques and machine learning.

## 🎯 Objective
- ✅ Build a spam detection model using **Multinomial Naive Bayes**.
- ✅ Apply **NLP techniques** to extract meaningful features from text messages.
- ✅ Classify messages as **spam (unwanted)** or **ham (legitimate)**.
- ✅ Evaluate model performance using **accuracy, precision, recall, and F1-score**.

## 📂 Dataset Information
- **Dataset Source:** SMS Spam Collection
- **Number of Records:** `5,572` messages
- **Columns:**
  - `labels`: Classification label ('spam' or 'ham')
  - `messages`: The text content to be classified

### 📊 Dataset Distribution
```python
messages.labels.value_counts()
```
| Label  | Count |
|--------|-------|
| Ham    | 4825  |
| Spam   | 747   |

---

## 🛠 Data Preprocessing
### 🔹 Steps Involved:
- **Regex Cleaning:** Remove unwanted characters and special symbols.
- **Stopwords Removal:** Eliminate common words using NLTK's stopwords.
- **Stemming:** Convert words to their root forms using **Porter Stemmer**.

```python
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

ps = PorterStemmer()
corpus = []

for i in range(len(messages)):
    message = re.sub(r'[^a-zA-Z]', ' ', messages['messages'][i])
    message = message.lower().split()
    message = [ps.stem(word) for word in message if word not in stopwords.words('english')]
    corpus.append(' '.join(message))
```

---

## 🔢 Feature Extraction
### 1️⃣ Bag of Words (BOW)
Convert text into numerical format using **CountVectorizer**.
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
```

### 2️⃣ TF-IDF (Term Frequency-Inverse Document Frequency)
Another method to convert text into numerical format.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=2500)
X_tf = tf.fit_transform(corpus).toarray()
```

---

## 🤖 Model Training
### 📌 Using **Multinomial Naive Bayes** for Spam Classification
```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
spam_detect_model = MultinomialNB().fit(x_train, y_train)
y_pred = spam_detect_model.predict(x_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
```
### ✅ Accuracy with Bag of Words: **98.56%**
### ✅ Accuracy with TF-IDF: **97.93%**

---

## 📊 Model Evaluation Metrics
| Metric      | BOW Model | TF-IDF Model |
|------------|----------|-------------|
| **Accuracy**  | 98.56%   | 97.93%       |
| **Precision** | High     | High        |
| **Recall**    | High     | High        |

---

## 🚀 Conclusion
- The **Multinomial Naive Bayes algorithm** effectively classifies text messages as spam or ham.
- **BOW and TF-IDF both perform well**, with **BOW achieving slightly higher accuracy**.
- The model can be **used in real-world applications** such as **email filtering and SMS spam detection**.

## 🔮 Future Improvements
🔹 **Enhance feature extraction:** Use **Word Embeddings (Word2Vec, FastText)**.  
🔹 **Try other models:** Test **Logistic Regression, SVM, or Random Forest**.  
🔹 **Deploy the model:** Integrate into a **real-time spam filtering system**.

---

## 📁 Repository Structure
| File Name              | Description                          |
|------------------------|--------------------------------------|
| `SpamDetection.ipynb`  | Jupyter Notebook with model code    |
| `SMSSpamCollection`    | Dataset file                        |
| `README.md`            | Project documentation               |
| `requirements.txt`     | Dependencies for running the model  |

---

## 🛠 Technologies Used
- **Python** (pandas, sklearn, nltk, matplotlib)
- **Machine Learning** (Naive Bayes)
- **Natural Language Processing (NLP)**
- **Data Preprocessing & Visualization**

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

📩 **For queries, reach out via [LinkedIn](https://www.linkedin.com/in/mokhit-khan-9234622b3/)**

---

**🚀 Thanks for exploring this project!**
