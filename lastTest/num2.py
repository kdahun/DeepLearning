import numpy as np
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical

class Spam:
    def __init__(self):
        docs = ['분노', '공포', '기쁨', '슬픔', '혐오']
        labels = np.array([0, 1, 2, 3, 4])

        t = Tokenizer()
        t.fit_on_texts(docs)

        encoded_docs = t.texts_to_sequences(docs)

        vocab_size = len(t.word_index) + 1
        max_length = 3

        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        labels_one_hot = to_categorical(labels)  # Convert labels to one-hot encoding

        self.model = Sequential()
        self.model.add(Embedding(vocab_size, 8, input_length=max_length))
        self.model.add(Flatten())
        self.model.add(Dense(5, activation='softmax'))  # Use softmax activation for multi-class classification
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(padded_docs, labels_one_hot, epochs=50, verbose=1)

        loss, accuracy = self.model.evaluate(padded_docs, labels_one_hot, verbose=1)

        print('정확도 =', accuracy)
        print('loss =', loss)

        test_docs = ['공포', '혐오']

        encoded_test_docs = t.texts_to_sequences(test_docs)
        padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_length, padding='post')

        predicted_probabilities = self.model.predict(padded_test_docs)
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        label_mapping = {0: '분노', 1: '공포', 2: '기쁨', 3: '슬픔', 4: '혐오'}

        predicted_categories = [label_mapping[label] for label in predicted_labels]

        print('Predicted categories:', predicted_categories)

Spam = Spam()
