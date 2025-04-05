import os
import io
import tarfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Reshape, Dropout, Concatenate
from tensorflow.keras.models import Sequential, Model
import pretty_midi
import matplotlib.pyplot as plt
import librosa
    
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
        x = Dense(np.prod(input_shape),  activation='sigmoid')(x)
        x = Reshape(input_shape)(x)
        processed.append(x)

    # Merge the tracks into a single piano roll
    output = Concatenate(axis=-1)(processed)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Extrack piano roll of an instrument
def extract_instrument_roll(pm, program_range, drum=False):
    # Initialize the piano roll to prevent "None" to be returned.
    roll = np.zeros((512, 128), dtype=float)
    # Look for correct track and extract piano roll
    for inst in pm.instruments:
        if drum != inst.is_drum:
            continue
        if drum or inst.program in program_range:
            inst_roll = inst.get_piano_roll(fs=16).T[:512, :128] / 127.0  # Normalize velocity to [0, 1]
            roll = np.maximum(roll, inst_roll)
    return roll

# Function to load all preprocessed Lakh MIDI data in batch
def load_lakh_data(file_list, batch_size):
    num_files = len(file_list)
    batch_count = 0

    # Infinite loop for generator
    while True:
        for i in range(0, num_files, batch_size):
            batch_files = file_list[i:i + batch_size]
            tracks_batch = []
            print(f"\nLoading batch {batch_count + 1}...")

            for file in batch_files:
                if file is not None:
                    try:
                        # Define appropriate program range and extract piano roll
                        pm = pretty_midi.PrettyMIDI(file)
                        drum = extract_instrument_roll(pm, [], drum=True)
                        bass = extract_instrument_roll(pm, range(32, 40))
                        pad = extract_instrument_roll(pm, range(0, 8))

                        # Initialize the piano roll to prevent "None" to be returned.
                        melody_roll = np.zeros((512, 128), dtype=float)

                        melodic_programs = [
                            range(40, 48), range(48, 56), range(56, 64),
                            range(64, 72), range(72, 80), range(80, 88), range(88, 96)
                        ]
                        # Randomize possible melody instruments for higher variety
                        np.random.shuffle(melodic_programs)

                        for prog in melodic_programs:
                            melody_roll = extract_instrument_roll(pm, prog)
                            # Select first non-empty piano roll from randomized list
                            if melody_roll.sum() > 0:
                                break

                        # Skip current file if all instruments are empty
                        if drum.sum() == 0 and bass.sum() == 0 and pad.sum() == 0 and melody_roll.sum() == 0:
                            print(f"Skipped empty file: {file}")
                            continue

                        print(f"Loaded file: {os.path.basename(file)}")
                        # Append the piano roll of four tracks to the batch
                        tracks_batch.append([drum, bass, pad, melody_roll])
                    except Exception as e:
                        print(f"Failed to process {file}: {e}")
                        continue

            if len(tracks_batch) == 0:
                print(f"Batch {batch_count + 1} is empty, skipping...")
                continue
            batch_count += 1
            # Transpose "list of samples" to "list of tracks"
            batch = list(zip(*tracks_batch))
            inputs = [np.array(track) for track in batch]
            output = np.concatenate(inputs, axis=-1)
            yield tuple(inputs), output

# Function to plot training history
def plot_training_history(history, path):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()

    plt.savefig(path, dpi=300)
    plt.show()


# Paths
musegan_save_path = "../../trained_model/musegan"
trained_musegan_path = "../../trained_model/musegan.h5"
lakh_dataset_path = "../../../dataset/Extracted/Lakh"
result_plot_path = "../../../Result/Performance/performance.png"

# Load Extracted Lakh MIDI data
file_list = [os.path.join(lakh_dataset_path, f) for f in os.listdir(lakh_dataset_path) if f.endswith(".mid")]
np.random.shuffle(file_list)
split_index = int(len(file_list) * 0.8)
train_files, valid_files = file_list[:split_index], file_list[split_index:]

# Define model parameters
batch_size = 32
epochs = 20
steps_per_epoch = len(train_files) // batch_size
validation_steps = len(valid_files) // batch_size

# Preprocess data and form batches
train_batch = load_lakh_data(train_files, batch_size)
valid_batch = load_lakh_data(valid_files, batch_size)

# train_batch = tf.data.Dataset.from_generator(
#     lambda: load_lakh_data(train_files, batch_size),
#     output_signature=(
#         tuple(tf.TensorSpec(shape=(None, 512, 128), dtype=tf.float32) for _ in range(4)),  # ← wrapped in tuple
#         tf.TensorSpec(shape=(None, 512, 512), dtype=tf.float32)
#     )
# )
#
# valid_batch = tf.data.Dataset.from_generator(
#     lambda: load_lakh_data(valid_files, batch_size),
#     output_signature=(
#         tuple(tf.TensorSpec(shape=(None, 512, 128), dtype=tf.float32) for _ in range(4)),
#         tf.TensorSpec(shape=(None, 512, 512), dtype=tf.float32)
#     )
# )

# Check if trained model already exists
# if os.path.exists(trained_musegan_path):
# if False:
#     musegan = tf.keras.models.load_model(trained_musegan_path)
#     print("MuseGAN model successfully loaded.")
#
# else:
print("MuseGAN model not found, building MuseGAN model...")
# Build MuseGAN Model
print("MuseGAN model summary:")
musegan = build_musegan()
musegan.summary()

# Train MuseGAN Model
print("MuseGAN train start...")
lakh_history = musegan.fit(
    train_batch,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=valid_batch,
    validation_steps=validation_steps
)

plot_training_history(lakh_history, result_plot_path)

musegan.save(musegan_save_path)

# print("MuseGAN validation start...")
# musegan_valid_loss = musegan.evaluate(lakh_valid_data, steps=lakh_validation_steps)
# print(f"MuseGAN validation loss: {musegan_valid_loss}")
# MuseGAN Validation Loss: 0.059657011181116104

###################
# Generate Output #
###################

def generate_piano_roll(musegan, sequence_length=512, pitch_range=128):
    """
    Generate a new piano roll using MuseGAN.
    """
    # Create random input for MuseGAN
    random_input = np.random.rand(1, sequence_length, pitch_range)

    # Generate piano roll
    piano_roll = musegan.predict(random_input)

    # Binarize the output (convert to binary piano roll)
    piano_roll = (piano_roll > 0.5).astype(int)
    return piano_roll

def visualize_piano_roll(piano_roll, count, save_path=None):
    """
    Visualizes a piano roll using a heatmap.

    Parameters:
        piano_roll (numpy.ndarray): The piano roll to visualize (shape: [timesteps, pitches]).
        save_path (str): Optional path to save the visualization as an image file.
        count (int): count-th piano roll image
    """
    if len(piano_roll.shape) == 3:
        piano_roll = piano_roll[0]  # Extract if batch dimension is present

    plt.figure(figsize=(10, 6))
    plt.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='Greys', interpolation='nearest')
    plt.colorbar(label="Note Intensity")
    plt.xlabel("Time Steps")
    plt.ylabel("MIDI Pitches")
    plt.title("Generated Piano Roll")
    plt.tight_layout()

    if save_path:
        filename = os.path.join(save_path, f"{count:03d}.png")
        plt.savefig(filename, dpi=300)
        print(f"Piano roll visualization saved as '{filename}'.")
    else:
        plt.show()
    plt.close()

# while True:
#     # Generate piano roll
#     print("Generating symbolic music using Simple MuseGAN...")
#     generated_piano_roll = generate_piano_roll(musegan)
#     visualize_piano_roll(generated_piano_roll)
#
#     answer = input("Continue? (y/n) ")
#     if answer == "y":
#         continue
#     break

# # Generate piano roll
# midi_path = "../Result/midi"
# image_path = "../Result/image"
# for i in range(0, 30):
#     print("Generating symbolic music using Simple MuseGAN...")
#     generated_piano_roll = generate_piano_roll(musegan)
#     midi_filename = os.path.join(midi_path, f"{i:03d}.npy")
#     np.save(midi_filename, generated_piano_roll)
#     visualize_piano_roll(generated_piano_roll, i, image_path)

