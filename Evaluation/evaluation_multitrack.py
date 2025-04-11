import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

# Constants
STEPS_PER_BAR = 32
NUM_BARS = 16
NUM_TRACKS = 4
TRACK_NAMES = ['drum', 'bass', 'pad', 'lead']
PITCH_CLASSES = 128

# Metric 1: Empty Bar Rate
def empty_bar_rate(piano_roll):
    eb_rates = []
    for i in range(NUM_TRACKS):
        track = piano_roll[i]
        bars = np.split(track, NUM_BARS, axis=1)
        empty_bars = [np.sum(bar) == 0 for bar in bars]
        eb_rates.append(np.mean(empty_bars))
    return eb_rates


# Metric 2: Used Pitch Classes
def used_pitch_classes(piano_roll):
    upc_scores = []
    for i in range(NUM_TRACKS):
        if i == 0:  # Skip drum
            upc_scores.append(None)
            continue
        track = piano_roll[i]
        bars = np.split(track, NUM_BARS, axis=1)
        pitch_classes = []
        for bar in bars:
            active = np.any(bar, axis=1)
            pcs = set([pitch % 12 for pitch in np.where(active)[0]])
            pitch_classes.append(len(pcs))
        upc_scores.append(np.mean(pitch_classes))
    return upc_scores


# Metric 3: Qualified Notes
def qualified_notes(piano_roll, min_duration=3):
    qn_scores = []
    for i in range(NUM_TRACKS):
        if i == 0: # Skip drum
            qn_scores.append(None)
            continue
        track = piano_roll[i]
        qualified = 0
        total = 0
        for pitch in range(PITCH_CLASSES):
            onsets = np.where(track[pitch] == 1)[0]
            if len(onsets) == 0:
                continue
            duration = 0
            prev = -2
            for t in onsets:
                if t == prev + 1:
                    duration += 1
                else:
                    if min_duration <= duration <= STEPS_PER_BAR:
                        qualified += 1
                    duration = 1
                prev = t
            if min_duration <= duration <= STEPS_PER_BAR:
                qualified += 1
            total += len(onsets)
        qn_scores.append(qualified / total if total > 0 else 0)
    return qn_scores


# Metric 4: Drum Pattern Consistency
def drum_pattern(piano_roll):
    track = piano_roll[0]  # drum
    bars = np.split(track, NUM_BARS, axis=1)
    pattern_scores = []
    for bar in bars:
        steps = np.sum(bar, axis=0)
        binary_pattern = (steps > 0).astype(int)
        grid = np.tile([1 if i % 4 == 0 else 0 for i in range(STEPS_PER_BAR)], len(binary_pattern) // STEPS_PER_BAR)
        match = np.sum(binary_pattern == grid)
        pattern_scores.append(match / len(binary_pattern))
    return [np.mean(pattern_scores)] + [None] * (NUM_TRACKS - 1)


# Metric 5: Tonal Distance
def tonal_distance(piano_roll):
    pitch_class_hist = lambda track: np.sum(track.reshape(128, -1), axis=1).reshape(-1)[:128]
    normalize = lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x

    td_matrix = np.zeros((NUM_TRACKS, NUM_TRACKS))
    for i in range(1, NUM_TRACKS):  # skip drum
        for j in range(i + 1, NUM_TRACKS):
            if i == 0 or j == 0:
                continue
            pc_i = np.zeros(12)
            pc_j = np.zeros(12)
            for pitch in range(128):
                pc_i[pitch % 12] += piano_roll[i][pitch].sum()
                pc_j[pitch % 12] += piano_roll[j][pitch].sum()
            pc_i = normalize(pc_i)
            pc_j = normalize(pc_j)
            td = cosine(pc_i, pc_j)
            td_matrix[i, j] = td
    return td_matrix


# Master function to evaluate one piano roll
def evaluate_piano_roll(piano_roll):
    piano_roll = np.stack([
        piano_roll["drum"].T,
        piano_roll["bass"].T,
        piano_roll["pad"].T,
        piano_roll["lead"].T
    ])

    eb = empty_bar_rate(piano_roll)
    upc = used_pitch_classes(piano_roll)
    qn = qualified_notes(piano_roll)
    dp = drum_pattern(piano_roll)

    metrics_table = pd.DataFrame([eb, upc, qn, dp], index=['Empty Bar Rate', 'Used Pitch Classes', 'Qualified Notes', 'Drum Pattern Consistency'], columns=TRACK_NAMES)
    td_matrix = tonal_distance(piano_roll)
    td_df = pd.DataFrame(td_matrix, columns=TRACK_NAMES, index=TRACK_NAMES)

    return metrics_table, td_df


def evaluate_folder(folder_path, label='generated output'):
    metric_tables = []
    td_matrices = []

    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    print(f"Found {len(npy_files)} files in '{folder_path}'.")

    count = 0
    for filename in npy_files:
        path = os.path.join(folder_path, filename)
        try:
            piano_roll = np.load(path, allow_pickle="True").item()
            metrics_table, td_df = evaluate_piano_roll(piano_roll)
            metric_tables.append(metrics_table)
            td_matrices.append(td_df)
            count += 1
            if count % 1000 == 0:
                print(f"Calculated {count} files.")
        except Exception as e:
            print(f"Failed to process {filename}: {type(e).__name__}: {e}")

    # Average the metric tables
    combined_metrics = pd.concat(metric_tables).groupby(level=0).mean()
    combined_td = sum(td_matrices) / len(td_matrices)

    print(f"\n===== AVERAGED SYMBOLIC METRICS for '{label}' =====\n")
    print(combined_metrics)

    print(f"\n===== AVERAGED TONAL DISTANCE MATRIX for '{label}' =====\n")
    print(combined_td)

    return combined_metrics, combined_td

train_folder_path = "../../../dataset/Preprocessed/Lakh/MultiTrack"
gen_folder_path = "../Result/npy"

# evaluate_folder(train_folder_path, label="Training Data")
evaluate_folder(gen_folder_path, label="Generated Output")
