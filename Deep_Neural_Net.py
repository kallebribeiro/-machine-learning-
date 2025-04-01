import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Hyperparameter definition
INPUT_SHAPE = 20  # Number of columns in the dataset (features)
HIDDEN_LAYERS = [128, 256, 128, 64, 32]  # Hidden layers with the number of neurons
DROPOUT_RATE = 0.3  # Dropout rate to prevent overfitting
LEARNING_RATE = 0.001  # Learning rate
EPOCHS = 100  # Maximum number of epochs for training
BATCH_SIZE = 32  # Batch size for weight updates

# Generating a synthetic dataset to simulate a real-world problem
np.random.seed(42)
x_train = np.random.rand(1000, INPUT_SHAPE)  # 1000 samples, 20 features
x_test = np.random.rand(200, INPUT_SHAPE)  # 200 test samples
y_train = np.random.randint(0, 2, size=(1000, 1))  # Binary classes (0 or 1)
y_test = np.random.randint(0, 2, size=(200, 1))

# Defining the deep neural network model
model = Sequential()

# First hidden layer with Batch Normalization to stabilize training
model.add(Dense(HIDDEN_LAYERS[0], activation='relu', input_shape=(INPUT_SHAPE,)))
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_RATE))

# Adding hidden layers with regularization
for units in HIDDEN_LAYERS[1:]:
    model.add(Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT_RATE))

# Output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compiling the model with Adam optimizer and binary cross-entropy loss function
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

# Defining callbacks for Early Stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Training the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, checkpoint]
)

# Final model evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
