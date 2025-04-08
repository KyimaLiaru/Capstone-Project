import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Reshape, Dropout, Concatenate
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
import librosa

##################################################
# Load Lakh Midi Dataset and Train MuseGAN Model #
##################################################

# Function to load all preprocessed Lakh MIDI data in batch
def load_lakh_data(file_list, batch_size):
    """
    Generator to load Lakh MIDI data in batches.
    """
    num_files = len(file_list)

    while True:  # Infinite loop for generator
        for i in range(0, num_files, batch_size):
            batch_files = file_list[i:i + batch_size]
            batch_data = np.array([np.load(file, allow_pickle=True).item() for file in batch_files])

            drum = np.array([d["drum"] for d in batch_data])
            bass = np.array([d["bass"] for d in batch_data])
            keyboard = np.array([d["piano"] for d in batch_data])
            synth_lead = np.array([d["melody"] for d in batch_data])
            input_tracks = [drum, bass, keyboard, synth_lead]

            yield input_tracks, np.concatenate([track for track in input_tracks], axis=-1)

# Building Simplified MuseGAN Model
def build_musegan(input_shape=(96, 128), num_tracks=4):
    """
    Build a simple MuseGAN model for symbolic music generation.
    """
    inputs = [Input(shape=input_shape) for _ in range(num_tracks)]
    processed = []

    for inp in inputs:
        x = LSTM(128, return_sequences=True)(inp)
        x = Dropout(0.3)(x)
        x = LSTM(128, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        x = Dense(128 * 96,  activation='sigmoid')(x)
        x = Reshape((96, 128))(x)
        processed.append(x)

    output = Concatenate(axis=-1)(processed)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Function to plot training history
def plot_training_history(history):
    """
    Plots training and validation accuracy and loss.
    """
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
    plt.show()

# Paths
musegan_save_path = "../../trained_model/musegan"
trained_musegan_path = "../../trained_model/musegan.h5"
preprocessed_lakh_data_path = "../../../dataset/Preprocessed/Lakh/MultiTrack"

# Split dataset into training and validation sets
file_list = [os.path.join(preprocessed_lakh_data_path, f) for f in os.listdir(preprocessed_lakh_data_path) if
             f.endswith(".npy")]
np.random.shuffle(file_list)
split_index = int(len(file_list) * 0.8)
lakh_train_files, lakh_valid_files = file_list[:split_index], file_list[split_index:]

# Load preprocessed Lakh MIDI data and define model parameters
lakh_batch_size = 32
lakh_epochs = 20

lakh_train_data = load_lakh_data(lakh_train_files, lakh_batch_size)
lakh_valid_data = load_lakh_data(lakh_valid_files, lakh_batch_size)

lakh_steps_per_epoch = len(lakh_train_files) // lakh_batch_size
lakh_validation_steps = len(lakh_valid_files) // lakh_batch_size

# Check if trained model already exists
if os.path.exists(trained_musegan_path):
    musegan = tf.keras.models.load_model(trained_musegan_path)
    print("MuseGAN model successfully loaded.")

else:
    print("MuseGAN model not found, building MuseGAN model...")
    # Build MuseGAN Model
    print("MuseGAN model summary:")
    musegan = build_musegan()
    musegan.summary()

    # Train MuseGAN Model
    print("MuseGAN train start...")
    lakh_history = musegan.fit(
        lakh_train_data,
        steps_per_epoch=lakh_steps_per_epoch,
        epochs=lakh_epochs,
        batch_size=lakh_batch_size,
        validation_data=lakh_valid_data,
        validation_steps=lakh_validation_steps
    )

    plot_training_history(lakh_history)

    musegan.save(musegan_save_path)

# print("MuseGAN validation start...")
# musegan_valid_loss = musegan.evaluate(lakh_valid_data, steps=lakh_validation_steps)
# print(f"MuseGAN validation loss: {musegan_valid_loss}")
# MuseGAN Validation Loss: 0.059657011181116104

###############################################
# Load NSynth Dataset and Train WaveNet Model #
###############################################

def build_wavenet():
    """
    Build a simple WaveNet model for audio synthesis.
    """
    input_layer = Input(shape=(16000, 1))
    x = Conv1D(32, kernel_size=2, dilation_rate=1, activation="relu", padding="causal")(input_layer)
    for rate in [2, 4, 8, 16]:
        x = Conv1D(32, kernel_size=2, dilation_rate=rate, activation="relu", padding="causal")(x)
    x = Flatten()(x)
    output_layer = Dense(16000, activation="tanh")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def load_nsynth_data(file_list, batch_size):
    """
    Generator to load preprocessed NSynth .npy files in batches.
    """
    num_files = len(file_list)

    while True:
        for i in range(0, num_files, batch_size):
            batch_files = file_list[i:i + batch_size]
            batch_data = [np.load(file, allow_pickle=True).item() for file in batch_files]
            x_batch = np.array([data['audio'] for data in batch_data])
            x_batch = x_batch[..., np.newaxis]
            yield x_batch, x_batch

input_sample_rate = 16000

# Define paths
wavenet_save_path = "../../trained_model/wavenet"
trained_wavenet_path = "../../trained_model/wavenet.h5"
preprocessed_nsynth_data_path = f"../../../dataset/Preprocessed/NSynth/{input_sample_rate}"

temp = True
if temp:
    asdf = 0
else:
    if os.path.exists(trained_wavenet_path):
        musegan = tf.keras.models.load_model(trained_wavenet_path)
        print("WaveNet model successfully loaded")
    else:
        # Define batch size and sample rate for input data
        nsynth_batch_size = 32
        input_sample_rate = 16000

        print("WaveNet model not found, building WaveNet model...")
        nsynth_train_files = [
            os.path.join(preprocessed_nsynth_data_path, f)
            for f in os.listdir(preprocessed_nsynth_data_path)
            if f.startswith('train') and f.endswith('.npy')
        ]
        nsynth_valid_files = [
            os.path.join(preprocessed_nsynth_data_path, f)
            for f in os.listdir(preprocessed_nsynth_data_path)
            if f.startswith('valid') and f.endswith('.npy')
        ]
        nsynth_test_files = [
            os.path.join(preprocessed_nsynth_data_path, f)
            for f in os.listdir(preprocessed_nsynth_data_path)
            if f.startswith('test') and f.endswith('.npy')
        ]

        nsynth_train_data = load_nsynth_data(nsynth_train_files, nsynth_batch_size)
        print("NSynth train data loaded.")
        nsynth_valid_data = load_nsynth_data(nsynth_valid_files, nsynth_batch_size)
        print("NSynth validation data loaded.")
        nsynth_test_data = load_nsynth_data(nsynth_test_files, nsynth_batch_size)
        print("NSynth test data loaded.")

        print("WaveNet model summary")
        wavenet = build_wavenet()
        wavenet.summary()

        # Define model parameters
        steps_per_epoch = len(nsynth_train_files) // nsynth_batch_size
        validation_steps = len(nsynth_valid_files) // nsynth_batch_size
        test_steps = len(nsynth_test_files) // nsynth_batch_size

        # Train WaveNet
        print("WaveNet train start...")
        wavenet.fit(
            nsynth_train_data,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            validation_data=nsynth_valid_data,
            validation_steps=validation_steps
        )

        # Evaluate the model on the test set
        test_loss = wavenet.evaluate(nsynth_test_data, steps=test_steps)
        print(f"WaveNet test Loss: {test_loss}")

        wavenet.save(wavenet_save_path)

###################
# Generate Output #
###################

def generate_piano_roll(musegan, sequence_length=96, pitch_range=128):
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

