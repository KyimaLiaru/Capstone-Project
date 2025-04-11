import os

import numpy as np
import tensorflow as tf
import pretty_midi
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt

###################
# Generate Output #
###################

def generate_piano_roll(musegan, sequence_length=512, pitch_range=128):
    # Create random input for MuseGAN
    drum_input = np.random.rand(1, sequence_length, pitch_range)
    bass_input = np.random.rand(1, sequence_length, pitch_range)
    pad_input = np.random.rand(1, sequence_length, pitch_range)
    lead_input = np.random.rand(1, sequence_length, pitch_range)
    inputs = [drum_input, bass_input, pad_input, lead_input]

    # Generate piano roll
    drum_output, bass_output, pad_output, lead_output = musegan.predict(inputs)

    # Remove batch dimension
    drum_output = drum_output[0]
    bass_output = bass_output[0]
    pad_output = pad_output[0]
    lead_output = lead_output[0]

    # Binarize the output (convert to binary piano roll)
    drum_output = (drum_output > 0.5).astype(int)
    bass_output = (bass_output > 0.5).astype(int)
    pad_output = (pad_output > 0.5).astype(int)
    lead_output = (lead_output > 0.5).astype(int)

    return drum_output, bass_output, pad_output, lead_output


def visualize_piano_roll(drum, bass, pad, lead, save_path, count):
    os.makedirs(save_path, exist_ok=True)
    combined_path = os.path.join(save_path, f"combined_{count:02d}.png")
    grid_path = os.path.join(save_path, f"grid_{count:02d}.png")

    combined_roll = np.vstack([drum[:100].T, bass[:100].T, pad[:100].T, lead[:100].T])
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
        ax.set_title(f"{title} Piano Roll")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("MIDI Pitch")

    plt.tight_layout()
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"Saved grid roll to {grid_path}.")

def piano_roll_to_instrument(roll, program=0, is_drum=False):
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
    fs = 16  # frames per second
    time_step = 1.0 / fs
    for pitch in range(roll.shape[1]):
        active = False
        note_on = 0
        for t in range(roll.shape[0]):
            if roll[t, pitch] > 0 and not active:
                active = True
                note_on = t
            elif roll[t, pitch] == 0 and active:
                note_off = t
                start = note_on * time_step
                end = note_off * time_step
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
                active = False
        if active:
            start = note_on * time_step
            end = roll.shape[0] * time_step
            inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
    return inst

def save_tracks_to_midi(drum, bass, pad, lead, output_path, count=1):
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(piano_roll_to_instrument(drum, is_drum=True))
    midi.instruments.append(piano_roll_to_instrument(bass, program=np.random.choice(range(32, 40))))
    midi.instruments.append(piano_roll_to_instrument(pad,  program=np.random.choice(range(0, 8))))
    midi.instruments.append(piano_roll_to_instrument(lead, program=np.random.choice(range(40, 96))))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, f"midi_{count:02d}.mid")
    midi.write(output_file)
    print(f"Saved generated output to: {output_file}")

def save_tracks_to_npy(drum, bass, pad, lead, output_path, count=1):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, f"npy_{count:02d}.npy")
    dict = {
        "drum": drum.T,
        "bass": bass.T,
        "pad": pad.T,
        "lead": lead.T
    }
    np.save(output_file, dict)
    print(f"Saved generated output to: {output_file}")


def print_note_durations_from_track(track, track_name="Track", threshold=0.5):
    """
    Prints the duration (in time steps) of every note event in a single piano roll track.

    Args:
        track: NumPy array of shape (512, 128) or (128, 512)
        track_name: String label for the track
        threshold: Value above which a note is considered "on"
    """
    if track.shape[0] == 512 and track.shape[1] == 128:
        track = track.T  # Transpose to shape (128, 512)

    print(f"\n=== {track_name} Track ===")
    for pitch in range(128):
        active = False
        start = 0
        for t in range(track.shape[1]):
            is_on = track[pitch, t] > threshold
            if is_on and not active:
                active = True
                start = t
            elif not is_on and active:
                duration = t - start
                print(f"Pitch {pitch} duration: {duration}")
                active = False
        if active:
            duration = track.shape[1] - start
            print(f"Pitch {pitch} duration: {duration}")


# Load MuseGAN model
musegan_save_path = "../../trained_model/musegan-old/musegan_checkpoints/musegan_epoch_11.h5"
musegan_save_path_2 = "../../trained_model/musegan-old/musegan_checkpoints/musegan_epoch_16.h5"
# musegan_save_path_3 = "../../trained_model/musegan.h5"
musegan_old_path = "../../trained_model/musegan-old/musegan.h5"
figure_path = "../Result/Rolls"
midi_path = "../Result/MIDI"
npy_path = "../Result/npy"

# musegan = tf.keras.models.load_model(musegan_save_path)
# musegan2 = tf.keras.models.load_model(musegan_save_path_2)
# musegan3 = tf.keras.models.load_model(musegan_save_path_3)
print("Model successfully loaded.")
# for i in range(1, 5):
#     drum, bass, pad, lead = generate_piano_roll(musegan)
#     visualize_piano_roll(drum, bass, pad, lead, figure_path, i)
#     save_tracks_to_midi(drum, bass, pad, lead, midi_path, i)
# for i in range(5, 9):
#     drum, bass, pad, lead = generate_piano_roll(musegan2)
#     visualize_piano_roll(drum, bass, pad, lead, figure_path, i)
#     save_tracks_to_midi(drum, bass, pad, lead, midi_path, i)
# for i in range(9, 13):
#     drum, bass, pad, lead = generate_piano_roll(musegan3)
#     visualize_piano_roll(drum, bass, pad, lead, figure_path, i)
#     save_tracks_to_midi(drum, bass, pad, lead, midi_path, i)

musegan = tf.keras.models.load_model(musegan_old_path)
musegan2 = tf.keras.models.load_model(musegan_save_path)
musegan3 = tf.keras.models.load_model(musegan_save_path_2)
for i in range(1, 21):
    # musegan_save_path = f"../../trained_model/musegan_checkpoints/musegan_epoch_{i:02d}.h5"
    np.random.seed(i)
    drum, bass, pad, lead = generate_piano_roll(musegan)
    drum2, bass2, pad2, lead2 = generate_piano_roll(musegan2)
    drum3, bass3, pad3, lead3 = generate_piano_roll(musegan3)
    # print_note_durations_from_track(drum, "drum")
    # print_note_durations_from_track(bass3, "bass")
    print_note_durations_from_track(pad, "pad")
    # print_note_durations_from_track(lead2, "lead")
    break
    # visualize_piano_roll(drum, bass3, pad, lead2, figure_path, i)
    # save_tracks_to_midi(drum, bass3, pad, lead2, midi_path, i)
    # save_tracks_to_npy(drum, bass3, pad, lead2, npy_path, i)
