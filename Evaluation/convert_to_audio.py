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

    # Optional: assign default instrument (most MIDI files include their own)
    # fs.program_select(0, sfid, 0, 0)

    # Load and play the MIDI file (it renders into the .wav file)
    player = fs.midi_player_add(midi_file)
    fs.midi_player_play(player)

    # Wait until the MIDI is completely rendered to WAV
    while fs.midi_player_get_status(player) == fluidsynth.FLUID_PLAYER_PLAYING:
        time.sleep(0.1)

    # Clean up
    fs.delete()
    print(f"WAV file rendered to: {output_path}")


# === Example usage ===
midi_path = "your_generated.mid"
soundfont_path = "VintageDreamsWaves-v2.sf2"
output_wav_path = "output.wav"

midi_to_wav(midi_path, soundfont_path, output_wav_path)