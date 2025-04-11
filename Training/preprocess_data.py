import tensorflow as tf
import librosa
import tarfile
import os
import numpy as np
import pretty_midi
import json


#########################################
# Preprocess and Save Lakh Midi Dataset #
#########################################

# Extract piano roll of given instrument.
def extract_instrument_roll(pm, program_range, drum=False):
    roll = np.zeros((512, 128), dtype=float)
    for inst in pm.instruments:
        if drum != inst.is_drum:
            continue
        if drum:
            inst_roll = extract_drum_roll(inst)
            roll = np.maximum(roll, inst_roll)
        elif inst.program in program_range:
            inst_roll = inst.get_piano_roll(fs=16).T[:512, :128]
            inst_roll = (inst_roll > 0).astype(float)
            roll = np.maximum(roll, inst_roll)
    return roll

# Extract drum from the instrument
def extract_drum_roll(inst, fs=16, length=512):
    roll = np.zeros((length, 128), dtype=float)
    for note in inst.notes:
        if note.pitch < 0 or note.pitch > 127:
            continue
        start = int(note.start * fs)
        end = int(note.end * fs)
        start = max(0, min(start, length - 1))
        end = max(start + 1, min(end, length))
        roll[start:end, note.pitch] = 1.0
    return roll

# Main function to preprocess lakh data
def process_lakh_data(file):
    try:
        pm = pretty_midi.PrettyMIDI(file)
        pad = extract_instrument_roll(pm, range(0, 8))
        bass = extract_instrument_roll(pm, range(32, 40))
        drum = extract_instrument_roll(pm, [], drum=True)

        # Randomly select a melodic instrument
        melodic_programs = [range(40, 48), range(48, 56), range(56, 64), range(64, 72), range(72, 80), range(80, 88), range(88, 96)]
        np.random.shuffle(melodic_programs)
        lead = np.zeros((512, 128), dtype=float)
        for prog in melodic_programs:
            lead = extract_instrument_roll(pm, prog)
            if lead.sum() > 0:
                break

        if pad.sum() == 0 and bass.sum() == 0 and drum.sum() == 0 and lead.sum() == 0:
            return None
        return {
            "drum": drum,
            "bass": bass,
            "pad": pad,
            "lead": lead
        }
    except Exception as e:
        # print(f"Error processing file: {e}")
        return None

# Define the base path for the Lakh MIDI Dataset
lakh_dataset_path = "../../../dataset/Raw/Lakh/lmd_matched.tar.gz"
lakh_preprocess_output_path = "../../../dataset/Preprocessed/Lakh/MultiTrack"

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

            if os.path.exists(output_file):
                continue

            # Extract process the data from the tar.gz file
            file = tar.extractfile(member)
            if file is not None:
                result = process_lakh_data(file)
                # if np.sum(result["drum"]) == 0:
                #     print("No drum data in", file)

                if result is not None:
                    print(f"Processing {output_file}")
                    print(f'Pad Sum: {np.sum(result["pad"])}, {result["pad"].shape}')
                    print(f'Bass Sum: {np.sum(result["bass"])}, {result["bass"].shape}')
                    print(f'Drum Sum: {np.sum(result["drum"])}, {result["drum"].shape}')
                    print(f'Lead Sum: {np.sum(result["lead"])}, {result["lead"].shape}')
                    # Save the preprocessed data
                    np.save(output_file, result)
                    print(f"Saved {output_file}")
                    count += 1

            if count % 100 == 1:
                asdf = "asdf"
                print(f"Processed {count} files")
print("Preprocess Completed.")