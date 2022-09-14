# ai 적용 예측
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5])
y = x*2+1

print(x)
print(y)

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x,y,epochs=100, verbose=1)

print('y:',y,',predict:',model.predict(x).flatten())
