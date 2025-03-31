import tensorflow as tf
import librosa
import tarfile
import os
import numpy as np
import pretty_midi

#########################################
# Preprocess and Save Lakh Midi Dataset #
#########################################

# Function to process MIDI files
def process_lakh_data(file):
    """
    Process a MIDI file object and return its piano roll.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file)
        piano_roll = midi_data.get_piano_roll(fs=16)  # Quantize to 16 steps/second
        piano_roll = (piano_roll > 0).astype(int)  # Binary representation
        return piano_roll.T[:96, :128]  # Trim to 96 time steps and 128 pitches
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Define the base URL for the Lakh MIDI Dataset
lakh_dataset_path = "../Capstone_Project_Dataset/Raw/Lakh/lmd_matched.tar.gz"
lakh_preprocess_output_path = "../Capstone_Project_Dataset/Preprocessed/Lakh/output"

# Check whether the output directory exists
if not os.path.exists(lakh_preprocess_output_path):
    os.makedirs(lakh_preprocess_output_path)

# Open the tar.gz file and process each MIDI file
with tarfile.open(lakh_dataset_path, "r:gz") as tar:
    i = 1
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
                piano_roll = process_lakh_data(file)
                if piano_roll is not None:
                    print(f"Lakh Processed {member.name}, Shape: {piano_roll.shape}")
                    # Save the preprocessed data
                    np.save(lakh_output_path, piano_roll)
                    print(f"Lakh Saved: {lakh_output_path}")

######################################
# Preprocess and Save NSynth Dataset #
######################################

# Load and parse the nsynth.tfrecord dataset file
def parse_tfrecord(record):
    feature_description = {
        'audio': tf.io.FixedLenFeature([64000], tf.float32),  # 4 seconds of audio at 16000 Hz
        'instrument_family': tf.io.FixedLenFeature([], tf.int64),
        'pitch': tf.io.FixedLenFeature([], tf.int64),
        'velocity': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(record, feature_description)

def load_dataset(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return parsed_dataset

def process_nsynth_data(nsynth_file_path, nsynth_output_path, dataset_type, sample_rate):
    raw_dataset = tf.data.TFRecordDataset(nsynth_file_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord)

    count = 0
    for record in parsed_dataset:
        # Check whether the preprocessed data already exists
        output = os.path.join(nsynth_output_path, f"{dataset_type}_{count:08d}.npy")
        if os.path.exists(output):
            count += 1
            continue

        # Extract and preprocess data
        audio = record['audio'].numpy()
        audio = librosa.resample(audio, orig_sr=64000, target_sr=sample_rate)
        audio = audio / np.max(np.abs(audio))
        instrument_family = record['instrument_family'].numpy()
        pitch = record['pitch'].numpy()
        velocity = record['velocity'].numpy()

        # Combine into a dictionary
        preprocessed_data = {
            'audio': audio,
            'instrument_family': instrument_family,
            'pitch': pitch,
            'velocity': velocity,
        }

        # Save the preprocessed data
        np.save(output, preprocessed_data)
        count += 1

    print(f"Saved {count} preprocessed samples for {dataset_type} dataset.")

# Define file paths for local dataset
train_dataset_path = "../Capstone_Project_Dataset/Raw/NSynth/nsynth-train.tfrecord"
valid_dataset_path = "../Capstone_Project_Dataset/Raw/NSynth/nsynth-valid.tfrecord"
test_dataset_path = "../Capstone_Project_Dataset/Raw/NSynth/nsynth-test.tfrecord"

hz = 8000

# Define output path for preprocessed nsynth data
nsynth_preprocess_output_path = f"../Capstone_Project_Dataset/Preprocessed/NSynth/output/{hz}"

# Check whether the output directory exists
if not os.path.exists(nsynth_preprocess_output_path):
    os.makedirs(nsynth_preprocess_output_path)

# Preprocess and save datasets
process_nsynth_data(train_dataset_path, nsynth_preprocess_output_path, 'train', hz)
process_nsynth_data(valid_dataset_path, nsynth_preprocess_output_path, 'valid', hz)
process_nsynth_data(test_dataset_path, nsynth_preprocess_output_path, 'test', hz)
