import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import *

class myCallback(Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('loss')<0.03:
            print('\n Stop training.')
            self.model.stop_training=True

class Dnn:
    def getData(self):

        total_data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews']) # 판다스로 텍스트 파일 읽어오기
        total_data['label'] = np.select([total_data.ratings > 3], [1], default=0) # 별점이 3보다 높으면 1 낮으면 0 라벨 열에 값 넣어주기
        total_data.drop_duplicates(subset=['reviews'], inplace=True)  # reviews 열에서 중복인 내용이 있다면 중복 제거

        x = total_data['reviews'].values
        y = total_data['label'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)
        return x_train, x_test, y_train, y_test
    def Tokennizer(self,x_train, x_test):
        t = Tokenizer()
        t.fit_on_texts(x_train)
        encode=t.texts_to_sequences(x_train)
        print(t.word_index)
        print(len(t.word_index)+1)

        max=0
        for i in encode:
            if max<len(i):
                max=len(i)

        print(max)
        print(len(encode))
        x_train = t.texts_to_sequences(x_train)





        t2 = Tokenizer()
        t2.fit_on_texts(x_test)
        t.texts_to_sequences(x_test)

        x_test = t.texts_to_sequences(x_test)


        return x_train,x_test

    def study(self,x_train, x_test, y_train, y_test):
        callbacks = myCallback()
        print(x_train)
        print(x_test)
        x_train = pad_sequences(x_train, maxlen=41)
        x_test = pad_sequences(x_test, maxlen=41)
        vocab_size = 308542

        model = Sequential()
        model.add(Embedding(vocab_size, 64, input_length=41))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, batch_size=60, epochs=20, verbose=1
                            ,validation_data=(x_test, y_test),callbacks=[callbacks])

        result = model.evaluate(x_test, y_test, verbose=1)

        print(result)

        a = model.predict(x_test)

        model.save('obs_model.h5')
        new_obs_model = keras.models.load_model('obs_model.h5')
        b = list()
        for i in a:
            if i > 0.5:
                b.append(1)
            else:
                b.append(0)

        t = 0
        f = 0
        for i, j, in zip(b, y_test):
            if i == j:
                t = t + 1
            else:
                f = f + 1

        print("참 : ", t)
        print("거짓 : ", f)




dnn=Dnn()
x_train, x_test, y_train, y_test=dnn.getData()
x_train,x_test=dnn.Tokennizer(x_train, x_test)
dnn.study(x_train, x_test, y_train, y_test)






