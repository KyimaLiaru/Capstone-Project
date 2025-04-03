import tensorflow as tf
import librosa
import tarfile
import os
import numpy as np
import pretty_midi

#########################################
# Preprocess and Save Lakh Midi Dataset #
#########################################

def extract_instrument_roll(pm, program_range, drum=False):
    roll = np.zeros((96, 128), dtype=float)
    for inst in pm.instruments:
        if drum != inst.is_drum:
            continue
        if drum or inst.program in program_range:
            inst_roll = inst.get_piano_roll(fs=16).T[:96, :128] / 127.0  # Normalize velocity to [0, 1]
            roll = np.maximum(roll, inst_roll)
    return roll

def process_lakh_data(file):
    try:
        pm = pretty_midi.PrettyMIDI(file)
        piano_roll = extract_instrument_roll(pm, range(0, 8))
        bass_roll = extract_instrument_roll(pm, range(32, 40))
        drum_roll = extract_instrument_roll(pm, [], drum=True)

        # Randomly select a melodic instrument
        melodic_programs = [range(40, 48), range(48, 56), range(56, 64), range(64, 72), range(72, 80), range(80, 88), range(88, 96)]
        np.random.shuffle(melodic_programs)
        melody_roll = None
        for prog in melodic_programs:
            melody_roll = extract_instrument_roll(pm, prog)
            if melody_roll.sum() > 0:
                break
            melody_roll = np.zeros((96, 128), dtype=float)

        if piano_roll.sum() == 0 and bass_roll.sum() == 0 and drum_roll.sum() == 0 and melody_roll.sum() == 0:
            return None
        return {
            "piano": piano_roll,
            "bass": bass_roll,
            "drum": drum_roll,
            "melody": melody_roll
        }
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Define the base path for the Lakh MIDI Dataset
lakh_dataset_path = "../../../../dataset/Raw/Lakh/lmd_matched.tar.gz"
lakh_preprocess_output_path = "../../../../dataset/Preprocessed/Lakh/MultiTrack"

# Check whether the output directory exists
if not os.path.exists(lakh_preprocess_output_path):
    os.makedirs(lakh_preprocess_output_path)

# Open the tar.gz file and process each MIDI file
with tarfile.open(lakh_dataset_path, "r:gz") as tar:
    count = 0
    for member in tar.getmembers():
        if member.name.endswith(".mid"):
            # Set the output file name
            prefix_parts = member.name.split("/")  # e.g., 'A', 'B', etc.
            prefix = "".join(prefix_parts[1:4])  # three directory levels, typically alphabet letters
            basename = os.path.basename(member.name).replace(".mid", ".npy")
            output_file = os.path.join(lakh_preprocess_output_path, f"{prefix}_{basename}")

            # Extract process the data from the tar.gz file
            file = tar.extractfile(member)
            if file is not None:
                result = process_lakh_data(file)
                if result is not None:
                    # Save the preprocessed data
                    np.save(output_file, result)
                    print(f"Saved {output_file}")
                    count += 1

            if count % 100 == 0:
                print(f"Processed {count} files")