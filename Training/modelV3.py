import os
import io
import re
import json
import time
import sys
import tarfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Reshape, Dropout, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import AUC, Precision, Recall
import pretty_midi
import matplotlib.pyplot as plt
import librosa
import pandas as pd


##################################################
# Load Lakh Midi Dataset and Train MuseGAN Model #
##################################################

# Building Simplified MuseGAN Model
def build_musegan(input_shape=(512, 128), num_tracks=4):
    inputs = [Input(shape=input_shape) for _ in range(num_tracks)]
    processed = []

    # Process each track independently
    for inp in inputs:
        x = LSTM(256, return_sequences=True)(inp)
        x = Dropout(0.3)(x)
        x = LSTM(256, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        x = Dense(np.prod(input_shape), activation='sigmoid')(x)
        x = Reshape(input_shape)(x)
        processed.append(x)

    # Merge the tracks into a single piano roll
    # output = Concatenate(axis=-1)(processed)
    model = Model(inputs=inputs, outputs=processed)
    model.compile(optimizer="adam", loss=["binary_crossentropy"]*4,
                  loss_weights=[1.0, 1.2, 0.5, 1.5],
                  metrics=[AUC(name="auc"), Precision(), Recall()]
    )
    return model

# Function to load all preprocessed Lakh MIDI data in batch
def load_data_from_directory(data_dir, file_list, batch_size):
    while True:
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i + batch_size]
            batch_inputs = []

            for file in batch_files:
                try:
                    data = np.load(os.path.join(data_dir, file), allow_pickle=True).item()
                    batch_inputs.append([data["drum"], data["bass"], data["pad"], data["lead"]])
                except Exception as e:
                    print(f"Failed to load {file}: {e}")
                    continue

            batch = list(zip(*batch_inputs))
            inputs = [np.array(track) for track in batch]
            outputs = [np.array(track) for track in batch]
            yield inputs, outputs

# Function to plot AUC and loss per track
def plot_training_history(csv_path, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 10))

    # Plot each track AUC
    plt.subplot(2, 1, 1)
    track_labels = ["drum", "bass", "pad", "lead"]
    for i, label in enumerate(track_labels):
        auc_key = 'reshape_auc' if i == 0 else f'reshape_{i}_auc'
        val_auc_key = 'val_reshape_auc' if i == 0 else f'val_reshape_{i}_auc'

        if auc_key in df.columns:
            plt.plot(df[auc_key], label=f'{label.capitalize()} AUC')
        if val_auc_key in df.columns:
            plt.plot(df[val_auc_key], linestyle='--', label=f'{label.capitalize()} Val AUC')

    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Track-wise AUC')
    plt.legend()

    # Plot each track loss
    plt.subplot(2, 1, 2)
    for i, label in enumerate(track_labels):
        loss_key = 'reshape_loss' if i == 0 else f'reshape_{i}_loss'
        val_loss_key = 'val_reshape_loss' if i == 0 else f'val_reshape_{i}_loss'

        if loss_key in df.columns:
            plt.plot(df[loss_key], label=f'{label.capitalize()} Loss')
        if val_loss_key in df.columns:
            plt.plot(df[val_loss_key], linestyle='--', label=f'{label.capitalize()} Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Track-wise Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved training AUC and Loss history to {save_path}")
    plt.show()
    plt.close()

# Function to plot average AUC and loss
def plot_average_history(csv_path, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 5))

    auc_cols = ['reshape_auc'] + [f'reshape_{i}_auc' for i in range(1, 4)]
    val_auc_cols = ['val_reshape_auc'] + [f'val_reshape_{i}_auc' for i in range(1, 4)]
    df['avg_auc'] = df[auc_cols].mean(axis=1)
    df['avg_val_auc'] = df[val_auc_cols].mean(axis=1)

    loss_cols = ['reshape_loss'] + [f'reshape_{i}_loss' for i in range(1, 4)]
    val_loss_cols = ['val_reshape_loss'] + [f'val_reshape_{i}_loss' for i in range(1, 4)]
    df['avg_loss'] = df[loss_cols].mean(axis=1)
    df['avg_val_loss'] = df[val_loss_cols].mean(axis=1)

    # Plot average AUC
    plt.subplot(1, 2, 1)
    plt.plot(df['avg_auc'], label='Avg Train AUC')
    plt.plot(df['avg_val_auc'], linestyle='--', label='Avg Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Average AUC Across Tracks')
    plt.legend()

    # Plot average loss
    plt.subplot(1, 2, 2)
    plt.plot(df['avg_loss'], label='Avg Train Loss')
    plt.plot(df['avg_val_loss'], linestyle='--', label='Avg Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Loss Across Tracks')
    plt.legend()

    plt.tight_layout()
    avg_path = save_path.replace(".png", "_avg.png")
    plt.savefig(avg_path, dpi=300)
    print(f"Saved average training AUC and Loss history to {avg_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":

    # Paths
    musegan_save_path = "../../trained_model/musegan.h5"
    musegan_checkpoint_path = "../../trained_model/musegan_checkpoints"
    musegan_checkpoint_name = os.path.join(musegan_checkpoint_path, "musegan_epoch_{epoch:02d}.h5")
    musegan_log_path = "../../trained_model/musegan_training_log.csv"
    trained_musegan_path = "../../trained_model/musegan.h5"
    training_history_path = "../../trained_model/training_history.json"
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    # lakh_dataset_path = "../../../dataset/Preprocessed/Lakh/MultiTrack-ver3.tar.gz"
    lakh_data_path = "../../../dataset/Preprocessed/Lakh/MultiTrack"
    result_plot_path = "../Result/Performance/performance.png"
    average_result_plot_path = "../Result/Performance/average_performance.png"

    # Define model parameters
    batch_size = 32
    epochs = 20

    try:
        if os.path.exists(training_history_path):
            with open(training_history_path, "r") as f:
                history = json.load(f)
            print("loaded previous training history.")

        file_list = [f for f in os.listdir(lakh_data_path) if f.endswith(".npy")]
        file_list.sort()

        split_index = int(len(file_list) * 0.8)
        train_files = file_list[:split_index]
        valid_files = file_list[split_index:]

        steps_per_epoch = len(train_files) // batch_size
        validation_steps = len(valid_files) // batch_size

        train_batch = load_data_from_directory(lakh_data_path, train_files, batch_size)
        valid_batch = load_data_from_directory(lakh_data_path, valid_files, batch_size)

        latest_epoch = 0
        latest_checkpoint = None

        if os.path.exists(musegan_checkpoint_path):
            checkpoint_files = [f for f in os.listdir(musegan_checkpoint_path) if re.match(r"musegan_epoch_(\d{2})\.h5", f)]
            checkpoint_epochs = [int(re.search(r"musegan_epoch_(\d{2})\.h5", f).group(1)) for f in checkpoint_files]
            latest_epoch = max(checkpoint_epochs)
            latest_checkpoint = os.path.join(musegan_checkpoint_path, f"musegan_epoch_{latest_epoch:02d}.h5")

        trained_model = False

        if os.path.exists(trained_musegan_path):
            musegan = tf.keras.models.load_model(trained_musegan_path)
            print("MuseGAN model successfully loaded.")
            trained_model = True
        elif latest_checkpoint and os.path.exists(latest_checkpoint):
            musegan = tf.keras.models.load_model(latest_checkpoint)
            print(f"MuseGAN model successfully loaded from checkpoint at epoch {latest_epoch}.")
        else:
            musegan = build_musegan()
            print("MuseGAN model not found, building new MuseGAN model...")

        if not trained_model:
            print("MuseGAN model summary:")
            musegan.summary()

            callbacks = [
                ModelCheckpoint(filepath=musegan_checkpoint_name, save_best_only=False, save_weights_only=False, verbose=1),
                CSVLogger(musegan_log_path, append=True)
            ]

            print("MuseGAN train start...")
            lakh_history = musegan.fit(
                train_batch,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=valid_batch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                initial_epoch=latest_epoch
            )
            musegan.save(musegan_save_path)

        plot_training_history(musegan_log_path, result_plot_path)
        plot_average_history(musegan_log_path, average_result_plot_path)

    except tf.errors.ResourceExhaustedError:
        print("OOM Error detected. Restarting training from last checkpoint...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
