import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import mlflow
import mlflow.tensorflow
import random
import matplotlib.pyplot as plt
import itertools
import pathlib

def define_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
batch_size = 32
img_height = 256
img_width = 256

def load_data_with_labels(data_dir):
    train_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    labels = []
    for images, batch_labels in train_data:
        labels.extend(batch_labels.numpy())  # Convert to NumPy and extend the list
    labels = np.array(labels)  # Convert list to NumPy array
    
    return train_data, val_data, labels

# Preprocessing function
def preprocess_data(dataset):
    """
    Normalize images and convert labels to categorical.
    """
    def process(image, label):
        # Normalize image
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, depth=25)
        return image, label

    # Apply normalization
    dataset = dataset.map(process)
    return dataset

def create_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def extract_labels(dataset):
    labels = []
    for batch in dataset:
        # Extract the labels from the batch
        _, batch_labels = batch
        labels.append(batch_labels.numpy())  # Convert to NumPy
    return np.concatenate(labels)  # Flatten into a 1D array

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(17, 17))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar(fraction=0.040)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.tight_layout()

def plot_classification_report(cr, title='Classification report', cmap='RdBu'):
    lines = cr.split('\n')
    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        if len(t) < 4 or not t[1].replace('.', '', 1).isdigit():
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1:4]]
        plotMat.append(v)

    plt.figure(figsize=(10, 15))
    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    x_tick_marks = ['precision', 'recall', 'f1-score']
    y_tick_marks = classes
    plt.xticks(np.arange(3), x_tick_marks, rotation=45, ha='right')
    plt.yticks(np.arange(len(classes)), y_tick_marks)
    
    thresh = np.max(plotMat) / 2.
    for i, j in itertools.product(range(len(classes)), range(3)):
        plt.text(j, i, format(plotMat[i][j], '.2f'),
                 horizontalalignment="center",
                 color="white" if plotMat[i][j] > thresh else "black")

    plt.ylabel('Classes', fontsize=15)
    plt.xlabel('Metrics', fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.model.save(self.filepath.format(epoch=epoch, **logs), overwrite=True)
            
def train_model(seed):
    define_seed(seed)
    
    # Load the dataset
    print("Loading Data...")
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    data_dir = os.path.join(script_dir, 'Dataset')
    train_data, val_data, labels = load_data_with_labels(data_dir)  # Returns training and validation datasets and labels
    print("Data Loaded")

    # Extract and preprocess labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Preprocess data
    train_data = preprocess_data(train_data)
    val_data = preprocess_data(val_data)
    encoded_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=25)
    label_dataset = tf.data.Dataset.from_tensor_slices(encoded_labels).batch(batch_size)
    print("Data preprocessed.")

    # Ensure datasets are properly batched
    train_dataset = train_data.unbatch().batch(32).prefetch(tf.data.AUTOTUNE)  # Adjust batch size as needed
    val_dataset = val_data.unbatch().batch(32).prefetch(tf.data.AUTOTUNE)

    print("Creating model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(25, activation='softmax')
    ])
    model.class_names = os.listdir(data_dir)
    print("Model created.")
    model.summary()
    print("Compiling model...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled.")

    # Create directory if it does not exist
    model_dir = f"../models/1.{seed}"
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_callback = CustomModelCheckpoint(
        filepath=f"../models/1.{seed}/best_model_{seed}.keras",
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
    with mlflow.start_run(run_name=f"Train_2.{seed}"):
        print("Training model...")
        history = model.fit(
            train_dataset,
            epochs=10,
            validation_data=val_dataset,
            callbacks=[checkpoint_callback, early_stopping_callback]
        )
        print("Model trained.")

        # Save training and validation accuracy of the best epoch
        best_epoch = checkpoint_callback.best_epoch
        train_accuracy = history.history['accuracy'][best_epoch]
        val_accuracy = history.history['val_accuracy'][best_epoch]
        with open(f"../models/1.{seed}/training_accuracy_{seed}.txt", "w") as f:
            f.write(f"Training accuracy: {train_accuracy}\n")
            f.write(f"Test accuracy: {val_accuracy}\n")

        model.load_weights(f"../models/1.{seed}/best_model_{seed}.keras")
        print("Best model loaded.")

        model.save(f"../models/1.{seed}/trained_model_{seed}.keras")
        print("Model saved.")

        # If you have a separate test dataset
        print("Evaluating model...")
        y_pred = []
        y_true = []
        for features, labels in val_dataset:
            predictions = model.predict(features)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=[str(cls) for cls in label_encoder.classes_])
        print("Model evaluated.")

        print("Saving classification report...")
        with open(f"../models/1.{seed}/classification_report_{seed}.txt", "w") as f:
            f.write(report)
        print("Classification report saved.")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=[str(cls) for cls in label_encoder.classes_])
        plt.savefig(f"../models/1.{seed}/confusion_matrix_{seed}.png")
        plt.close()

        # Classification Report Heatmap
        plot_classification_report(report)
        plt.savefig(f"../models/1.{seed}/classification_report_heatmap_{seed}.png")
        plt.close()

        # Log metrics and artifacts to mlflow
        mlflow.log_param("epochs", len(history.epoch))
        mlflow.log_metric("accuracy", accuracy)
        for epoch in range(len(history.epoch)):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_artifact(f"../models/1.{seed}/classification_report_{seed}.txt")
        mlflow.log_artifact(f"../models/1.{seed}/trained_model_{seed}.keras")
        mlflow.log_artifact(f"../models/1.{seed}/training_accuracy_{seed}.txt")
        mlflow.log_artifact(f"../models/1.{seed}/confusion_matrix_{seed}.png")
        mlflow.log_artifact(f"../models/1.{seed}/classification_report_heatmap_{seed}.png")

        print("\n\n\n")

        print("Training completed. Accuracy:", accuracy)
        print("Classification Report:\n", report)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5001")
    for i in range(8, 10):
        print(f"\n\nTraining model {i}...\n\n")
        train_model(seed=i)
        print(f"\n\nModel {i} trained.\n\n")