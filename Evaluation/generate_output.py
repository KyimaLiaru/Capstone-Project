import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt

###################
# Generate Output #
###################

def generate_piano_roll(musegan, sequence_length=512, pitch_range=128):
    """
    Generate a new piano roll using MuseGAN.
    """
    # Create random input for MuseGAN
    drum_input = np.random.rand(1, sequence_length, pitch_range)
    bass_input = np.random.rand(1, sequence_length, pitch_range)
    pad_input = np.random.rand(1, sequence_length, pitch_range)
    lead_input = np.random.rand(1, sequence_length, pitch_range)
    input = [drum_input, bass_input, pad_input, lead_input]

    # Generate piano roll
    output = musegan.predict(input)

    # Binarize the output (convert to binary piano roll)
    output = output[0]
    drum_output, bass_output, pad_output, lead_output = np.split(output, 4, axis=-1)

    # piano_roll = (piano_roll > 0.5).astype(int)
    return drum_output, bass_output, pad_output, lead_output


def visualize_piano_roll(drum, bass, pad, lead, save_path):
    """
    Visualizes a piano roll using a heatmap.

    Parameters:
        piano_roll (numpy.ndarray): The piano roll to visualize (shape: [timesteps, pitches]).
        save_path (str): Optional path to save the visualization as an image file.
        count (int): count-th piano roll image
    """

    os.makedirs(save_path, exist_ok=True)
    combined_path = os.path.join(save_path, "combined.png")
    grid_path = os.path.join(save_path, "grid.png")

    combined_roll = np.vstack([drum.T, bass.T, pad.T, lead.T])
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_roll, aspect="auto", origin="lower", cmap="hot")
    plt.title("Combined Piano Roll")
    plt.xlabel("Time Step")
    plt.ylabel("MIDI Pitch")
    plt.tight_layout()
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"Saved combined roll to {combined_path}.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    rolls = [drum, bass, pad, lead]
    titles = ["Drum", "Bass", "Pad", "Lead"]

    for ax, roll, title in zip(axes.flat, rolls, titles):
        ax.imshow(roll.T, aspect='auto', origin='lower', cmap='gray_r')
        ax.title(f"{titles} Piano Roll")
        ax.xlabel("Time Step")
        ax.ylabel("MIDI Pitch")

    plt.tight_layout()
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"Saved grid roll to {grid_path}.")

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

# Load MuseGAN model
musegan_save_path = "../../trained_model/musegan_checkpoints/musegan_epoch_15.h5"
figure_path = "../Result/Rolls"

musegan = tf.keras.models.load_model(musegan_save_path)

drum, bass, pad, lead = generate_piano_roll(musegan)
visualize_piano_roll(drum, bass, pad, lead, figure_path)