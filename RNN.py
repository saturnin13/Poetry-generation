import keras
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, Activation

model_lstm = Sequential()
# model_lstm.add(Embedding(input_dim = max_words, output_dim = 256, input_length = max_phrase_len))
# model_lstm.add(SpatialDropout1D(0.3))
model_lstm.add(LSTM(128, dropout = 0.3, recurrent_dropout = 0.3))
# model_lstm.add(Dense(256, activation = 'relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Activation('softmax'))
# model_lstm.add(Dense(5, activation = 'softmax'))

model_lstm.summary()

model_lstm.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

history = model_lstm.fit(
    X_train,
    y_train,
    validation_split = 0.1,
    epochs = 8,
    batch_size = 128
)

