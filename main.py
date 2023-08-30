import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()

train.isnull().sum()

train.info()

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
train['Product ID']=encoder.fit_transform(train['Product ID'])
train['Type']=encoder.fit_transform(train['Type'])

train.info()

y=train['Machine failure']
X=train.drop('Machine failure',axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)


"""# Neural Net"""

model=tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
