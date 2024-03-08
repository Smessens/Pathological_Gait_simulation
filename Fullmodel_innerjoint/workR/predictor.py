import numpy as np
import os
os.system("clear")
os.system("ls workR/fitness_data/")

dt = 1000e-7
tf = 5
F_max_alpha=0
v_max_alpha=0



name="workR/fitness_data/validationV3_tf"+str(tf)

print(str(name)+"memory_fitness.npy")




memory_fitness = np.load(str(name)+"memory_fitness.npy", allow_pickle=True).tolist() 
memory_fitness_breakdown = np.load(str(name)+"memory_fitness_breakdown.npy", allow_pickle=True).tolist() 
memory_suggestion = np.load(str(name)+"memory_suggestion.npy", allow_pickle=True).tolist()

memory_fitness = np.array(memory_fitness) 
memory_fitness_breakdown = np.array(memory_fitness_breakdown) 
memory_suggestion = np.array(memory_suggestion) 


print(memory_fitness_breakdown.shape)


# Define initial parameter values
initial_G_VAS = 2e-4
initial_G_SOL = 1.2 / 4000
initial_G_GAS = 1.1 / 1500
initial_G_TA = 1.1
initial_G_SOL_TA = 0.0001
initial_G_HAM = 2.166666666666667e-04
initial_G_GLU = 1 / 3000.
initial_G_HFL = 0.5
initial_G_HAM_HFL = 4
initial_G_delta_theta = 1.145915590261647

initial_theta_ref = 0.104719755119660
initial_k_swing = 0.26
initial_k_p = 1.909859317102744
initial_k_d = 0.2
initial_phi_k_off = 2.967059728390360

# Offset parameters with default values
initial_loff_TA =  0.72
initial_lopt_TA =  0.06
initial_loff_HAM = 0.85
initial_lopt_HAM = 0.10
initial_loff_HFL = 0.65
initial_lopt_HFL = 0.11


new = [0.00021650737025777133, 0.0003790779144082624, 0.0010310904239078675, 1.538052275586236, 0.00015463560386906556, 0.00017999407946658733, 0.0002936982267177814, 0.4849557161899837, 4.269665654011495, 1.2436758704624877, 0.1204244334553288, 0.26512676646138217, 2.870742081664975, 0.28665466207356327, 2.4682746982349038, 0.951227560132939, 0.05209916868257586, 0.8788646454861523, 0.07014205372407391, 0.48353091921693925, 0.17223210566037392]

print("GVAS", new[0]/initial_G_VAS)
print("GSOL", new[1]/initial_G_SOL)
print("GGAS", new[2]/initial_G_GAS)
print("GTA", new[3]/initial_G_TA)
print("GSOL_TA", new[4]/initial_G_SOL_TA)
print("GHAM", new[5]/initial_G_HAM)
print("GGLU", new[6]/initial_G_GLU)
print("GHFL", new[7]/initial_G_HFL)
print("GHAM_HFL", new[8]/initial_G_HAM_HFL)
print("G_delta_theta", new[9]/initial_G_delta_theta)
print("Theta_ref", new[10]/initial_theta_ref)
print("K_swing", new[11]/initial_k_swing)
print("K_p", new[12]/initial_k_p)
print("K_d", new[13]/initial_k_d)
print("Phi_k_off", new[14]/initial_phi_k_off)
print("Loff_TA", new[15]/initial_loff_TA)
print("Lopt_TA", new[16]/initial_lopt_TA)
print("Loff_HAM", new[17]/initial_loff_HAM)
print("Lopt_HAM", new[18]/initial_lopt_HAM)
print("Loff_HFL", new[19]/initial_loff_HFL)
print("Lopt_HFL", new[20]/initial_lopt_HFL)



import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers


X = memory_suggestion
y = memory_fitness

# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)

# Creating a Sequential model
model = Sequential()

# Adding layers to the model with dropout regularization and L1/L2 regularization
model.add(Dense(1000, activation='relu', input_shape=(21,), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1))  # Output layer

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

if 'history' in locals():
    del history


for i in range(100):
  # Training the model and storing the history
  print(i)
  if 'history' in locals():
      temp_history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
      history.history['loss'].extend(temp_history.history['loss'])
      history.history['val_loss'].extend(temp_history.history['val_loss'])
  else:
      history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


  import matplotlib.pyplot as plt

  # Plotting loss history
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss History')
  plt.legend()
  plt.show()

  # Evaluate the model
  loss = model.evaluate(X_val, y_val)
  print("Validation Loss:", loss)


  # Predicting on validation data
  y_pred = model.predict(X_val)

  # Plotting expected vs. true values
  plt.figure(figsize=(8, 6))

  y_pred = model.predict(X_train)
  plt.scatter(y_train, y_pred)
  
  y_pred = model.predict(X_val)
  plt.scatter(y_val, y_pred)


  plt.plot(np.arange(50),np.arange(50))

  plt.xlabel('True Values')
  plt.ylabel('Predicted Values')
  plt.title('Expected vs. True Values (Validation)')
  plt.grid(True)
  plt.show()