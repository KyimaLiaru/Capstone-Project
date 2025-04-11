import os.path
import time
import fluidsynth


def midi_to_wav(midi_file, soundfont_file, output_path):
    # Create a FluidSynth object with WAV file output only
    fs = fluidsynth.Synth(samplerate=44100)

    # Start FluidSynth in file-rendering mode
    fs.start(driver="file", filename=output_path)

    # Load the SoundFont
    sfid = fs.sfload(soundfont_file)

    # Load and play the MIDI file (it renders into the .wav file)
    player = fs.midi_player_add(midi_file)
    fs.midi_player_play(player)

    # Wait until the MIDI is completely rendered to WAV
    while fs.midi_player_get_status(player) == fluidsynth.FLUID_PLAYER_PLAYING:
        time.sleep(0.1)

    # Clean up
    fs.delete()
    print(f"WAV file rendered to: {output_path}")

def convert_folder(mid_folder, soundfont_file, wav_output_folder):
    if not os.path.exists(wav_output_folder):
        os.makedirs(wav_output_folder)

    midi_files = [f for f in os.listdir(mid_folder) if f.lower().endswith('.mid')]

    for midi_file in midi_files:
        midi_path = os.path.join(mid_folder, midi_file)
        wav_filename = os.path.splitext(midi_file)[0] + ".wav"
        wav_output_path = os.path.join(wav_output_folder, wav_filename)
        midi_to_wav(midi_path, soundfont_file, wav_output_path)

# Define paths
midi_path = "../Result/MIDI"
output_wav_path = "../Result/wav"
soundfont_path = "../Resource/Vintage Dreams Waves v2.sf2"

convert_folder(midi_path, soundfont_path, output_wav_path)