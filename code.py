import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load your dataset (Assume you have a CSV file 'reviews.csv' with 'review' and 'sentiment' columns)
df = pd.read_csv('reviews.csv')

# Check the first few rows of the dataset
print(df.head())

# Preprocess text: Clean and tokenize the reviews
reviews = df['review'].values
sentiments = df['sentiment'].values

# Encode the sentiment labels
label_encoder = LabelEncoder()
sentiments = label_encoder.fit_transform(sentiments)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

# Initialize the tokenizer and fit on the training data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert text data to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input size
max_len = 100  # Maximum length of each review (in words)
X_train_padded = pad_sequences(X_train_sequences, padding='post', maxlen=max_len)
X_test_padded = pad_sequences(X_test_sequences, padding='post', maxlen=max_len)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(SpatialDropout1D(0.2))  # Dropout to prevent overfitting
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=64, validation_data=(X_test_padded, y_test), verbose=2)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test data
score, accuracy = model.evaluate(X_test_padded, y_test, verbose=2)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Function to predict sentiment for a new review
def predict_sentiment(review):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, padding='post', maxlen=max_len)
    prediction = model.predict(padded)
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    return sentiment

# Test the model on new data
new_review = "This product is amazing! Highly recommended."
print(f"Sentiment: {predict_sentiment(new_review)}")
