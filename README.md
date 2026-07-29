# PhishGuard - Phishing Email Detection using Machine Learning

## Overview
PhishGuard is a Machine Learning-based web application that detects whether an email is **Phishing** or **Legitimate**. The application uses Natural Language Processing (NLP) and a trained Machine Learning model to analyze email text and predict its category.

## Features
- Detects phishing emails.
- User-friendly web interface.
- Machine Learning-based prediction.
- Fast and accurate classification.
- Stores prediction history in a SQLite database.

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- SQLite

## Project Structure

```
PhishGuard/
│── app.py
│── train_model.py
│── requirements.txt
│── database.db
│── phishing_email.csv
│── phishing_model.pkl
│── vectorizer.pkl
│── static/
│── templates/
```

## Machine Learning Algorithm
- TF-IDF Vectorization
- Multinomial Naive Bayes Classifier

## Installation

### Clone Repository
```bash
git clone https://github.com/prabha299/PhishGuard.git
cd PhishGuard
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python app.py
```

The application will run at:

```
http://127.0.0.1:5000
```

## Dataset
The project uses a phishing email dataset stored in:

```
phishing_email.csv
```

## How It Works
1. User enters email content.
2. Text is cleaned and transformed using TF-IDF Vectorizer.
3. The trained Machine Learning model predicts whether the email is:
   - Phishing
   - Legitimate
4. The prediction is displayed to the user.

## Future Improvements
- URL analysis
- Email attachment scanning
- Deep Learning models
- User authentication
- Real-time email detection

## Author
**Seran**

## License
This project is developed for educational purposes.
