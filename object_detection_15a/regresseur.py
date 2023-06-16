import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers, utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from keras import regularizers


data_with_onehot_encoding = pd.read_csv('/home/user/Documents/selected_row_data_and_onehot_encoded_columns.csv')

quantitative_feature_columns = ['length_x_object', 'width_y_object', 'height_z_object']
categorial_feature_columns = ['obj_wordnet_word_bag','obj_wordnet_word_bottle', 'obj_wordnet_word_candle','obj_wordnet_word_cap','obj_wordnet_word_glass','obj_wordnet_word_glasses',
                              'obj_wordnet_word_hat', 'obj_wordnet_word_laptop','obj_wordnet_word_pencil_case','obj_wordnet_word_phone',
                              'obj_wordnet_word_shoe','obj_wordnet_word_speaker','obj_wordnet_word_teddy_bear']
label_columns = ['weight_object']


scaler = StandardScaler()
scaled_quantitative_feature = scaler.fit_transform(data_with_onehot_encoding[quantitative_feature_columns])

categorial_feature = np.array(data_with_onehot_encoding[categorial_feature_columns].values)

Y = np.array(data_with_onehot_encoding[['weight_object']].values).astype(float)


# Division des données en ensembles "train", "validation" et "test"
X_quantitative_train_val, X_quantitative_test, X_categorical_train_val, X_categorical_test, Y_train_val, Y_test = train_test_split(
    scaled_quantitative_feature, categorial_feature, Y, test_size=0.15)

#X_quantitative_train, X_quantitative_val, X_categorical_train, X_categorical_val, Y_train, Y_val = train_test_split(
#    X_quantitative_train_val, X_categorical_train_val, Y_train_val, test_size=0.1)

print(len(X_quantitative_train_val))


def create_model():
  quantitative_input = Input(shape=(len(quantitative_feature_columns),))
  categorical_input = Input(shape=(len(categorial_feature_columns),))

  # the first branch operates on the quantitative_input input
  x = Dense(64, activation="relu")(quantitative_input)
  x = Dropout(0.15)(x)
  x = Dense(32, activation="relu")(x)
  x = Model(inputs=quantitative_input, outputs=x)

  # the second branch opreates on the second input
  y = Dense(26, activation="relu")(categorical_input)
  y = Dropout(0.15)(y)
  y = Dense(13, activation="relu")(y)
  y = Model(inputs=categorical_input, outputs=y)

  # combine the output of the two branches
  combined = concatenate([x.output, y.output])

  z = Dense(128, activation="relu")(combined)
  z = Dense(64, activation="relu")(z)
  z = Dropout(0.15)(z)
  z = Dense(32, activation="relu")(z)
  z = Dense(16, activation="relu")(z)
  z = Dense(1, activation="linear")(z)

  model = Model(inputs=[x.input, y.input], outputs=z)
  model.compile(optimizer=optimizers.Adam(learning_rate= 2e-3), loss='mean_squared_error')

  return model

model = create_model()

# history = model.fit([X_quantitative_train_val, X_categorical_train_val], Y_train_val, epochs=250,
#           validation_data=([X_quantitative_test, X_categorical_test], Y_test))

history = model.fit([scaled_quantitative_feature, categorial_feature], Y, epochs=250)

# Get the training info
loss     = history.history['loss']
#val_loss = history.history['val_loss']

# Visualize the history plots
plt.figure()
plt.plot(loss, 'b', label='Training loss')
#plt.plot(val_loss, 'm', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel("loss : MSE")
plt.xlabel("Epochs")
plt.legend()
plt.yscale("log")
plt.show()


score = model.evaluate([X_quantitative_test, X_categorical_test], Y_test)
print('Score de perte sur l\'ensemble test:', score)

'''
print("\nPredict : \n")

index = 8

Xq_to_pred = np.array([X_quantitative_test[index]])
Xc_to_pred = np.array([X_categorical_test[index]])

print(Xq_to_pred)
print(Xc_to_pred)
print(Y_test)

print(type(X_quantitative_test), X_quantitative_test.shape)

Y_pred = model.predict([X_quantitative_test, X_categorical_test], verbose='None')
print('Prediction: \n', Y_pred)

'''

save = input("to save enter 'y'") 
if save=='y':
  model.save('../keras_regresseur_saves/keras_regresseur_3_13_2/')


#print(model.summary())

#utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)