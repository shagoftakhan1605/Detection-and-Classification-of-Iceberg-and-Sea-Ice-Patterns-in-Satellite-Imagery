import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load datasets
ni_6s_data = np.load('NI_6s.npy')  # Replace with your actual path
ow_6s_data = np.load('OW_6s.npy')  # Replace with your actual path

# Assuming NI_6s.npy represents sea ice (label 0) and OW_6s.npy represents icebergs (label 1)
sea_ice_labels = np.zeros(ni_6s_data.shape[0])  # Label 0 for sea ice
iceberg_labels = np.ones(ow_6s_data.shape[0])   # Label 1 for icebergs

# Combine data and labels
X = np.concatenate((ni_6s_data, ow_6s_data), axis=0)
y = np.concatenate((sea_ice_labels, iceberg_labels), axis=0)

# Reshape data if necessary
X = X.reshape(X.shape[0], 128, 128, 1)  # Assuming images are 128x128 and grayscale

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up data generators
datagen = ImageDataGenerator()

train_generator = datagen.flow(X_train, y_train, batch_size=32)
validation_generator = datagen.flow(X_val, y_val, batch_size=32)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Single output unit for binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[early_stopping, model_checkpoint]
)

# Load the best model
model.load_weights('best_model.h5')

# Evaluate on validation data
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test Accuracy: {test_acc}')

# Predictions
Y_pred = model.predict(validation_generator, validation_generator.samples // validation_generator.batch_size + 1)
y_pred = np.round(Y_pred).astype(int).flatten()

# Confusion matrix and classification report
print('Confusion Matrix')
conf_matrix = confusion_matrix(y_val, y_pred)
print(conf_matrix)

print('Classification Report')
target_names = ['Sea Ice', 'Iceberg']
print(classification_report(y_val, y_pred, target_names=target_names))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', Validation'], loc='upper left')
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True)
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Simulate a classified image
classified_image = np.random.choice([0, 1], size=(128, 128), p=[0.7, 0.3])

# Visualize the classified image
plt.figure(figsize=(6, 6))
cmap = ListedColormap(['#87CEEB', '#FFFFFF'])  # Light blue for sea ice, white for iceberg
plt.imshow(classified_image, cmap=cmap, interpolation='nearest')
plt.title('Simulated Classified Image: Sea Ice (0) vs Iceberg (1)', fontsize=14)
cbar = plt.colorbar(ticks=[0, 1], fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(['Sea Ice (0)', 'Iceberg (1)'])
plt.clim(-0.5, 1.5)
plt.grid(False)
plt.show()
