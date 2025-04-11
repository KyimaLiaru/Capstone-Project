import os.path
import time
from midi2audio import FluidSynth

def convert_folder(midi_folder, soundfont_file, wav_output_folder):
    if not os.path.exists(wav_output_folder):
        os.makedirs(wav_output_folder)

    fs = FluidSynth(sound_font=soundfont_file)

    midi_files = [f for f in os.listdir(midi_folder) if f.lower().endswith('.mid')]

    for midi_file in midi_files:
        midi_path = os.path.join(midi_folder, midi_file)
        wav_filename = os.path.splitext(midi_file)[0] + ".wav"
        wav_output_name = os.path.join(wav_output_folder, wav_filename)

        fs.midi_to_audio(midi_path, wav_output_name)
        print(f"WAV file rendered to: {wav_output_name}")


# Define paths
midi_path = "../Result/MIDI"
wav_output_path = "../Result/wav"
soundfont_path = "../Resource/Vintage Dreams Waves v2.sf2"

convert_folder(midi_path, soundfont_path, wav_output_path)