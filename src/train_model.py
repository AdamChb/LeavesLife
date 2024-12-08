import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.tensorflow
import random

def define_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_data(data_dir, max_files_per_folder=1000):
    data = []
    labels = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label = folder
            files = os.listdir(folder_path)
            if len(files) > max_files_per_folder:
                files = random.sample(files, max_files_per_folder)
            for file in files:
                file_path = os.path.join(folder_path, file)
                image = tf.keras.preprocessing.image.load_img(file_path, target_size=(256, 256))
                image = tf.keras.preprocessing.image.img_to_array(image)
                data.append(image)
                labels.append(label)
    return np.array(data), np.array(labels)

def preprocess_data(X, y):
    X = X / 255.0
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = tf.keras.utils.to_categorical(y)
    return X, y, label_encoder

def create_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def train_model(seed):
    define_seed(seed)
    
    print("Loading data...")
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    data_dir = os.path.join(script_dir, 'Dataset')
    X, y = load_data(data_dir)
    X, y, label_encoder = preprocess_data(X, y)
    print("Data loaded.")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = create_dataset(X_train, y_train)
    test_dataset = create_dataset(X_test, y_test)
    print("Data splitted.")

    print("Creating model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.class_names = os.listdir(data_dir)
    print("Model created.")

    print("Compiling model...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled.")
    
    checkpoint_callback = ModelCheckpoint(
        filepath=f"../models/{seed}/best_model_{seed}.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    mlflow.set_experiment("LeavesLife")
    with mlflow.start_run(run_name=f"Train_{seed}"):
        print("Training model...")
        history = model.fit(train_dataset,
                            epochs=10,
                            validation_data=test_dataset,
                            callbacks=[checkpoint_callback, early_stopping_callback])
        print("Model trained.")
        
        model.load_weights(f"../models/{seed}/best_model_{seed}.keras")
        print("Best model loaded.")
        
        model.save(f"../models/{seed}/trained_model_{seed}.keras")
        print("Model saved.")

        print("Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_)
        print("Model evaluated.")
        
        print("Saving classification report...")
        with open(f"../models/{seed}/classification_report_{seed}.txt", "w") as f:
            f.write(report)
        print("Classification report saved.")

        mlflow.log_param("epochs", len(history.epoch))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_artifact(f"../models/{seed}/classification_report_{seed}.txt")
        mlflow.log_artifact(f"../models/{seed}/trained_model_{seed}.keras")
        
        print("\n\n\n")

        print("Training completed. Accuracy:", accuracy)
        print("Classification Report:\n", report)
    
if __name__ == "__main__":
    for i in range(0, 10):
        print(f"\n\nTraining model with seed {i}...\n\n")
        train_model(seed=i)
        print(f"\n\nModel trained with seed {i}.\n\n")