import os
import zipfile
import tensorflow as tf

# Extracting ZIP file
#Providing paths for zip file and extraction
zip_path = r"C:\Users\NANA\Downloads\archive (13).zip"
extract_path = r"C:\Users\NANA\Downloads\face_age_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Preparing image dataset 
image_dir = os.path.join(extract_path, 'face_age')

def parse_age_label(file_path):
    parts = tf.strings.split(file_path, os.sep)
    folder_name = parts[-2]
    age = tf.strings.to_number(folder_name, tf.float32)
    return age

def process_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    #Converts pixel values from [0, 255] to [-1, 1] as required by MobileNetV2
    return image

def load_image_and_age(file_path):
    image = process_image(file_path)
    age = parse_age_label(file_path)
    return image, age

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def augment(image, age):
    image = data_augmentation(image)
    return image, age

# Creating dataset
file_paths = tf.data.Dataset.list_files(os.path.join(image_dir, '*/*.png'), shuffle=True)

# Counting total samples
total_images = sum(1 for _ in file_paths)
train_size = int(0.8 * total_images)

# Reloading file_paths to reset iterator
file_paths = tf.data.Dataset.list_files(os.path.join(image_dir, '*/*.png'), shuffle=True)

full_dataset = file_paths.map(load_image_and_age, num_parallel_calls=tf.data.AUTOTUNE)

# Split dataset
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size)

# Applying augmentation only on training set
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print(f" Dataset prepared. Training: {train_size}, Validation: {total_images - train_size}")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Creating CNN model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet'
)

base_model.trainable = False  
# Freezing base for initial training

x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Early stopping callback
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=5,
    restore_best_weights=True,
    mode='min',
    verbose=1
)

#  Train (frozen base)
print("\n Training top layers (frozen base)...\n")
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[early_stopping_cb]
)

#  Fine-tune entire model
print("\n Fine-tuning entire model...\n")
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='mse', metrics=['mae'])

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'best_age_model_aug.h5',
    monitor='val_mae',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping_cb_ft = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=5,
    restore_best_weights=True,
    mode='min',
    verbose=1
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[checkpoint_cb, early_stopping_cb_ft]
)

print(" Training complete. Best model saved as 'best_age_model_aug.h5'")

# Evaluating the model on validation set
print("\n Evaluating model on validation set...")
model.load_weights('best_age_model_aug.h5')
loss, mae = model.evaluate(val_dataset)
print(f"Validation MSE: {loss:.2f}")
print(f"Validation MAE (Mean Absolute Error): {mae:.2f} years")


#Testing the model on a new image
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, 0)  
    return image

# test image path
test_image_path = r'C:\Users\NANA\Downloads\face_age_dataset\face_age\016\858.png'

# Preprocess and predict
img = load_and_preprocess_image(test_image_path)
predicted_age = model.predict(img)

print(f"Predicted Age: {predicted_age[0][0]:.2f} years")