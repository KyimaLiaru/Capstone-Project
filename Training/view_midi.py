import tarfile
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import io

def plot_multitrack_pianoroll(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)

    # Define instrument categories
    track_colors = {
        'Drums': 'Reds',
        'Piano': 'Blues',
        'Bass': 'Greens'
    }

    plt.figure(figsize=(12, 6))

    for inst in midi.instruments:
        # Get name
        name = 'Drums' if inst.is_drum else pretty_midi.program_to_instrument_name(inst.program)

        # Choose color
        if name.startswith("Acoustic Grand Piano"):
            cmap = track_colors.get('Piano', 'Greys')
        elif name.lower().find("bass") != -1:
            cmap = track_colors.get('Bass', 'Greys')
        elif inst.is_drum:
            cmap = track_colors.get('Drums', 'Greys')
        else:
            continue  # Skip other instruments

        # Extract piano roll
        roll = inst.get_piano_roll(fs=16)  # shape: [128, time]
        plt.imshow(
            roll,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            alpha=0.7
        )

    plt.xlabel("Time Steps")
    plt.ylabel("MIDI Pitches")
    plt.title("Multi-track Piano Roll")
    plt.colorbar(label="Note Intensity")
    plt.tight_layout()
    plt.show()

def plot_piano_pianoroll(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)

    plt.figure(figsize=(10, 5))
    piano_found = False

    for inst in midi.instruments:
        name = pretty_midi.program_to_instrument_name(inst.program).lower()

        if "piano" in name:
            piano_found = True
            roll = inst.get_piano_roll(fs=16)
            plt.imshow(
                roll,
                aspect='auto',
                origin='lower',
                cmap='Blues'
            )
            break

    if piano_found:
        plt.xlabel("Time Steps")
        plt.ylabel("MIDI Pitches")
        plt.title("Piano Piano Roll")
        plt.colorbar(label="Note Intensity")
        plt.tight_layout()
        plt.show()
    else:
        print("No piano instrument found in this MIDI file.")

def plot_drum_pianoroll(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)

    plt.figure(figsize=(10, 5))
    drum_found = False

    for inst in midi.instruments:
        if inst.is_drum:
            drum_found = True
            roll = inst.get_piano_roll(fs=16)
            plt.imshow(
                roll,
                aspect='auto',
                origin='lower',
                cmap='Reds'
            )
            break

    if drum_found:
        plt.xlabel("Time Steps")
        plt.ylabel("MIDI Drum Pitches")
        plt.title("Drum Piano Roll")
        plt.colorbar(label="Note Intensity")
        plt.tight_layout()
        plt.show()
    else:
        print("No drum track found in this MIDI file.")

def plot_bass_pianoroll(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)

    plt.figure(figsize=(10, 5))
    bass_found = False

    for inst in midi.instruments:
        name = pretty_midi.program_to_instrument_name(inst.program).lower()

        if "bass" in name:
            bass_found = True
            roll = inst.get_piano_roll(fs=16)  # shape: [128, time]
            plt.imshow(
                roll,
                aspect='auto',
                origin='lower',
                cmap='Greens'
            )
            break  # Only plot the first bass track found

    if bass_found:
        plt.xlabel("Time Steps")
        plt.ylabel("MIDI Pitches")
        plt.title("Bass Piano Roll")
        plt.colorbar(label="Note Intensity")
        plt.tight_layout()
        plt.show()
    else:
        print("No bass instrument found in this MIDI file.")

# Path to the tar.gz file
tar_path = "../../../../dataset/Raw/Lakh/lmd_matched.tar.gz"

count = 0

# Open the tar.gz archive
with tarfile.open(tar_path, "r:gz") as tar:
    # Find the first MIDI file
    for member in tar.getmembers():
        if member.name.endswith(".mid"):
            print(f"Found MIDI file: {member.name}")

            # Extract file-like object from the archive
            file_obj = tar.extractfile(member)
            if file_obj is not None:
                # Load with pretty_midi
                midi_data = pretty_midi.PrettyMIDI(io.BytesIO(file_obj.read()))

                if count == 0:
                    count += 1
                    continue

                print(f"Number of instruments: {len(midi_data.instruments)}\n")
                for i, inst in enumerate(midi_data.instruments):
                    name = pretty_midi.program_to_instrument_name(inst.program)
                    print(f"Instrument {i}: {name}")
                    print(f"  Program: {inst.program}")
                    print(f"  Is Drum: {inst.is_drum}")
                    print(f"  Number of notes: {len(inst.notes)}")

                # Visualize the first non-empty instrument
                for inst in midi_data.instruments:
                    if len(inst.notes) > 0:
                        piano_roll = inst.get_piano_roll(fs=16)
                        plt.figure(figsize=(12, 5))
                        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='gray_r')
                        plt.title("Piano Roll of First Non-Empty Instrument")
                        plt.xlabel("Time (frames)")
                        plt.ylabel("Pitch")
                        plt.colorbar(label='Velocity')
                        plt.tight_layout()
                        plt.show()
                        break
            print("\n")
            print("\n")
            print("\n")