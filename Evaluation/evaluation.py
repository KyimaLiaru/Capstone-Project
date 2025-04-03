import numpy as np
import os

def calculate_empty_bar_rate(piano_roll, time_steps_per_bar=16):
    """
    Calculate the Empty Bar Rate (EB).
    A bar is considered empty if no notes are played within the bar.
    """
    num_bars = piano_roll.shape[0] // time_steps_per_bar
    empty_bars = 0

    for i in range(num_bars):
        bar = piano_roll[i * time_steps_per_bar:(i + 1) * time_steps_per_bar]
        if np.sum(bar) == 0:
            empty_bars += 1

    return empty_bars / num_bars if num_bars > 0 else 0

def calculate_used_pitch_classes(piano_roll):
    """
    Calculate the Number of Used Pitch Classes (UPC).
    Count the distinct pitch classes used in the entire track.
    """
    pitch_classes = set()
    for time_step in piano_roll:
        active_pitches = np.where(time_step > 0)[0]
        pitch_classes.update(active_pitches % 12)  # Mod 12 for pitch classes
    return len(pitch_classes)

def calculate_qualified_note_rate(piano_roll, min_duration=3):
    """
    Calculate the Qualified Note Rate (QN).
    A note is qualified if it lasts at least `min_duration` time steps.
    """
    qualified_notes = 0
    total_notes = 0

    for col in range(piano_roll.shape[1]):
        active_time_steps = np.where(piano_roll[:, col] > 0)[0]
        if len(active_time_steps) > 0:
            note_durations = np.diff(np.hstack(([-1], active_time_steps, [piano_roll.shape[0]])))
            qualified_notes += np.sum(note_durations[:-1] >= min_duration)
            total_notes += len(active_time_steps)

    return qualified_notes / total_notes if total_notes > 0 else 0

def evaluate_musegan(directory_path, time_steps_per_bar=16):
    """
    Evaluate a collection of piano rolls with the selected metrics.
    """
    eb_scores = []
    upc_scores = []
    qn_scores = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".npy"):
            file = os.path.join(directory_path, file_name)

            piano_roll = np.load(file)

            eb = calculate_empty_bar_rate(piano_roll, time_steps_per_bar)
            upc = calculate_used_pitch_classes(piano_roll)
            qn = calculate_qualified_note_rate(piano_roll)

            eb_scores.append(eb)
            upc_scores.append(upc)
            qn_scores.append(qn)

            print(f"Evaluated {file_name}: EB={eb:.2f}, UPC={upc}, QN={qn:.2f}")

    return {
        "Empty Bar Rate (EB)": np.mean(eb_scores),
        "Used Pitch Classes (UPC)": np.mean(upc_scores),
        "Qualified Note Rate (QN)": np.mean(qn_scores)
    }

midi_path = "../../../Result/midi"
train_data_path = "../../../dataset/Lakh"

# Evaluate the piano rolls
results = evaluate_musegan(train_data_path)

# Print summary of results
print("\nSummary of Evaluation Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.2f}")


