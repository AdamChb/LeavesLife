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
    with mlflow.start_run(run_name=f"Train_1.{seed}"):
        print("Training model...")
        history = model.fit(train_dataset,
                            epochs=10,
                            validation_data=test_dataset,
                            callbacks=[checkpoint_callback, early_stopping_callback])
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

        print("Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_)
        print("Model evaluated.")
        
        print("Saving classification report...")
        with open(f"../models/1.{seed}/classification_report_{seed}.txt", "w") as f:
            f.write(report)
        print("Classification report saved.")

        # Confusion Matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plot_confusion_matrix(cm, classes=label_encoder.classes_)
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
    for i in range(0, 10):
        print(f"\n\nTraining model with seed {i}...\n\n")
        train_model(seed=i)
        print(f"\n\nModel trained with seed {i}.\n\n")