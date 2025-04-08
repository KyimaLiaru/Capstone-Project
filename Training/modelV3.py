import os
import io
import re
import json
import tarfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Reshape, Dropout, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
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
    output = Concatenate(axis=-1)(processed)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Function to load all preprocessed Lakh MIDI data in batch
def load_lakh_data(dataset_path, batch_size, data_type, split_ratio=0.8):
    # Infinite loop for generator
    print("Loading .tar.gz file...")
    with tarfile.open(dataset_path, "r:gz") as tar:
        files = [member for member in tar.getmembers() if member.name.endswith(".npy")]
        split_index = int(len(files) * split_ratio)

        file_list = files[:split_index] if data_type == "train" else files[split_index:]

        print(f"Total files in archive: {len(files)}")
        print(f"Using {len(file_list)} files for {data_type}")
        batch_count = 0
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i + batch_size]
            batch_inputs = []
            batch_count += 1
            print(f"\nLoading batch {batch_count}...")

            for file in batch_files:
                try:
                    ext_file = tar.extractfile(file)
                    buf = io.BytesIO(ext_file.read())
                    data = np.load(buf)

                    batch_inputs.append([data["drum"], data["bass"], data["pad"], data["lead"]])
                    print(f"Loaded file: {file}")

                except Exception as e:
                    print(f"Failed to process {file}: {e}")
                    continue

            if len(batch_inputs) == 0:
                # print(f"Batch {batch_count + 1} is empty, skipping...")
                continue

            # Transpose "list of samples" to "list of tracks"
            batch = list(zip(*batch_inputs))
            inputs = [np.array(track) for track in batch]
            output = np.concatenate(inputs, axis=-1)
            yield tuple(inputs), output

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
            output = np.concatenate(inputs, axis=-1)
            yield tuple(inputs), output

# Function to plot training history
def plot_training_history(csv_path, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df['accuracy'], label='Train Accuracy')
    plt.plot(df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(df['loss'], label='Train Loss')
    plt.plot(df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# Paths
musegan_save_path = "../../trained_model/musegan.h5"
musegan_checkpoint_path = "../../trained_model/musegan_checkpoints"
musegan_checkpoint_name = os.path.join(musegan_checkpoint_path, "musegan_epoch_{epoch:02d}.h5")
musegan_log_path = "../../trained_model/musegan_training_log.csv"
trained_musegan_path = "../../trained_model/musegan.h5"
training_history_path = "../../trained_model/training_history.json"
history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

if os.path.exists(training_history_path):
    with open(training_history_path, "r") as f:
        history = json.load(f)
    print("loaded previous training history.")

# lakh_dataset_path = "../../../dataset/Preprocessed/Lakh/MultiTrack-ver3.tar.gz"
lakh_data_path = "../../../dataset/Preprocessed/Lakh/MultiTrack"
result_plot_path = "../../../Result/Performance/performance.png"

# Define model parameters
batch_size = 32
epochs = 20

file_list = [f for f in os.listdir(lakh_data_path) if f.endswith(".npy")]
file_list.sort()

split_index = int(len(file_list) * 0.8)
train_files = file_list[:split_index]
valid_files = file_list[split_index:]

steps_per_epoch = len(train_files) // batch_size
validation_steps = len(valid_files) // batch_size

# Preprocess data and form batches
train_batch = load_data_from_directory(lakh_data_path, train_files, batch_size)
valid_batch = load_data_from_directory(lakh_data_path, valid_files, batch_size)

checkpoint_files = [f for f in os.listdir(musegan_checkpoint_path) if re.match(r"musegan_epoch_(\d{2})\.h5", f)]

latest_epoch = 0
latest_checkpoint = None

if checkpoint_files:
    checkpoint_epochs = [int(re.search(r"musegan_epoch_(\d{2})\.h5", f).group(1)) for f in checkpoint_files]
    latest_epoch = max(checkpoint_epochs)
    latest_checkpoint = os.path.join(musegan_checkpoint_path, f"musegan_epoch_{latest_epoch:02d}.h5")


trained_model = False

# Check if trained model already exists
if os.path.exists(trained_musegan_path):
# if False:
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
    # Build MuseGAN Model
    print("MuseGAN model summary:")
    musegan.summary()

    callbacks = [
        ModelCheckpoint(filepath=musegan_checkpoint_name, save_best_only=False, save_weights_only=False, verbose=1),
        CSVLogger(musegan_log_path, append=True)
    ]

    # Train MuseGAN Model
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




# print("MuseGAN validation start...")
# musegan_valid_loss = musegan.evaluate(lakh_valid_data, steps=lakh_validation_steps)
# print(f"MuseGAN validation loss: {musegan_valid_loss}")
# MuseGAN Validation Loss: 0.059657011181116104