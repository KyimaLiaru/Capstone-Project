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

def process_lakh_data(file):
    try:
        pm = pretty_midi.PrettyMIDI(file)
        pad = extract_instrument_roll(pm, range(0, 8))
        bass = extract_instrument_roll(pm, range(32, 40))
        drum = extract_instrument_roll(pm, [], drum=True)

        # Randomly select a melodic instrument
        melodic_programs = [range(40, 48), range(48, 56), range(56, 64), range(64, 72), range(72, 80), range(80, 88), range(88, 96)]
        np.random.shuffle(melodic_programs)
        lead = None
        for prog in melodic_programs:
            lead = extract_instrument_roll(pm, prog)
            if lead.sum() > 0:
                break
            lead = np.zeros((512, 128), dtype=float)

        if pad.sum() == 0 and bass.sum() == 0 and drum.sum() == 0 and lead.sum() == 0:
            return None
        return {
            "drum": drum,
            "bass": bass,
            "pad": pad,
            "lead": lead
        }
    except Exception as e:
        print(f"Error processing file: {e}")
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
                print(f"Processing {output_file}")
                print(f"Pad Sum: {np.sum(result['pad'])}")
                print(f"Bass Sum: {np.sum(result['bass'])}")
                print(f"Drum Sum: {np.sum(result['drum'])}")
                print(f"Lead Sum: {np.sum(result['lead'])}")
                # if np.sum(result["drum"]) == 0:
                #     print("No drum data in", file)

                if result is not None:
                    # Save the preprocessed data
                    # np.save(output_file, result)
                    # np.savez(output_file, drum=result["drum"], bass=result["bass"], pad=result["pad"], lead=result["lead"])
                    # print(f"Saved {output_file}")
                    count += 1

            if count % 100 == 1:
                print(f"Processed {count} files")

raise RuntimeError

######################################
# Preprocess and Save NSynth Dataset #
######################################

# Load and parse the nsynth.tfrecord dataset file
def parse_tfrecord(record):
    feature_description = {
        'audio': tf.io.FixedLenFeature([64000], tf.float32),
        'instrument_family': tf.io.FixedLenFeature([], tf.int64),
        'instrument_family_str': tf.io.FixedLenFeature([], tf.string),
        'pitch': tf.io.FixedLenFeature([], tf.int64),
        'velocity': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(record, feature_description)


def process_nsynth_data(file_path, output_base_path, dataset_type, sample_rate=16000):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed = raw_dataset.map(parse_tfrecord)

    counters = {key: 0 for key in TARGET_FAMILIES}
    for record in parsed:
        instrument_family_str = record['instrument_family_str'].numpy().decode('utf-8')
        instrument_source = record['instrument_source'].numpy()
        if instrument_family_str not in TARGET_FAMILIES:
            continue
        elif instrument_family_str == "bass" and instrument_source == 1:
            continue

        # # Determine instrument type based on family and source
        # instrument_family = record['instrument_family'].numpy()
        # if instrument_family == 0:
        #     instrument_type = "bass"
        # elif instrument_family == 4:
        #     instrument_type = "keyboard"
        # elif instrument_family == 9:
        #     instrument_type = "synth_lead"
        # else:
        #     continue  # Skip if not one of the desired types

        # Prepare save directory
        nsynth_preprocess_output_path = os.path.join(output_base_path, instrument_family_str)
        if not os.path.exists(nsynth_preprocess_output_path):
            os.makedirs(nsynth_preprocess_output_path)
        output_file = os.path.join(nsynth_preprocess_output_path,
                                   f"{dataset_type}_{counters[instrument_family_str]:08d}.npy")

        # Process audio
        audio = record['audio'].numpy()
        audio = librosa.resample(audio, orig_sr=64000, target_sr=sample_rate)
        audio = audio / np.max(np.abs(audio))

        # Save preprocessed data
        np.save(output_file, {
            'audio': audio,
            'instrument_family_str': instrument_family_str,
            'pitch': record['pitch'].numpy(),
            'velocity': record['velocity'].numpy(),
        })

        print(f"Saved {output_file}")
        counters[instrument_family_str] += 1

    for key, value in counters.items():
        print(f"Saved {value} samples as instrument {key}.")



# Target instrument families
TARGET_FAMILIES = {
    "bass": "bass",
    "keyboard": "keyboard",
    "synth_lead": "synth_lead"
}

nsynth_input_paths = {
    "train": "../../../../dataset/Raw/NSynth/nsynth-train.tfrecord",
    "valid": "../../../../dataset/Raw/NSynth/nsynth-valid.tfrecord",
    "test": "../../../../dataset/Raw/NSynth/nsynth-test.tfrecord"
}

nsynth_output_path = "../../../../dataset/Preprocessed/NSynth/16000"

for dtype, path in nsynth_input_paths.items():
    process_nsynth_data(path, nsynth_output_path, dtype, 16000)