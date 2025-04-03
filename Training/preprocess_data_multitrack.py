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
    roll = np.zeros((96, 128), dtype=int)
    for inst in pm.instruments:
        if drum != inst.is_drum:
            continue
        if drum or inst.program in program_range:
            inst_roll = inst.get_piano_roll(fs=16).T[:96, :128]
            roll = np.maximum(roll, (inst_roll > 0).astype(int))
    return roll

def process_lakh_data(file):
    try:
        pm = pretty_midi.PrettyMIDI(file)
        piano_roll = extract_instrument_roll(pm, range(0, 8))
        bass_roll = extract_instrument_roll(pm, range(32, 40))
        drum_roll = extract_instrument_roll(pm, [], drum=True)
        if piano_roll.sum() == 0 and bass_roll.sum() == 0 and drum_roll.sum() == 0:
            return None
        return {
            "piano": piano_roll,
            "bass": bass_roll,
            "drum": drum_roll
        }
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Define the base path for the Lakh MIDI Dataset
lakh_dataset_path = "../../../../dataset/Raw/Lakh/lmd_matched.tar.gz"
lakh_preprocess_output_path = "../../../../dataset/Preprocessed/Lakh/MultiTrackv2"

# Check whether the output directory exists
if not os.path.exists(lakh_preprocess_output_path):
    os.makedirs(lakh_preprocess_output_path)

# Open the tar.gz file and process each MIDI file
with tarfile.open(lakh_dataset_path, "r:gz") as tar:
    count = 0
    for member in tar.getmembers():
        if member.name.endswith(".mid"):
            # Check whether the preprocessed data exists
            lakh_output_filename = os.path.basename(member.name).replace(".mid", ".npy")
            lakh_output_path = os.path.join(lakh_preprocess_output_path, lakh_output_filename)
            if os.path.exists(lakh_output_path):
                continue

            # Extract process the data from the tar.gz file
            file = tar.extractfile(member)
            if file is not None:
                result = process_lakh_data(file)
                if result is not None:
                    # Save the preprocessed data
                    np.save(lakh_output_path, result)
                    print(f"Saved {lakh_output_path}")
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} files")