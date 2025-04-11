This project aims to implement artificial intelligence (AI) techniques, specifically Long Short-Term Memory (LSTM) networks in generating symbolic electronic music. Inspired by the existing MuseGAN model, a simplified version was developed to generate multi-track compositions focusing on melody, bass, piano (pads), and drums using Lakh MIDI Dataset, emphasizing on resource-efficient implementations. Although the original plan included symbolic-to-audio conversion with WaveNet, this was excluded due to lack of suitable MIDI-to-audio paired datasets and limited computational resources. Instead, thie project will focus on enhancing the symbolic generation model to produce multi-track compositions and utilizing external libraries, such as FluidSynth and SoundFonts, to render the generated output into audio.

Due to the dataset files being too large, here are the links for downloading the dataset files directly. The dataset used for this project are "Lakh MIDI Datset - Matched" and "Nsynth Dataset".

Lakh MIDI Dataset Download Link:
https://colinraffel.com/projects/lmd/

Being inspired by existing MuseGAN model, this project will be designing simplified versions of the model for suitable training time.

The folder structure should be designed as follows:
<pre>simiplified-musegan
├── dataset
│   ├── Preprocessed
│   │   └── Lakh 
│   │   │   └── MultiTrack (A folder where preprocessed files are saved)
│   └── Raw
│       └── Lakh (A folder where raw Lakh MIDI Dataset file should be saved)
│           └── lmd_matched.tar.gz
├── Model
│   ├── Capstone-Project(GitHub)
│   │   ├── Evaluation
│   │   │   ├── convert_to_audio.py
│   │   │   ├── evaluation.py
│   │   │   └── generate_output.py
│   │   ├── Training
│   │   │   ├── model.py
│   │   │   └── preprocess_data.py
│   │   ├── Resource (A folder where SoundFont .sf2 files shoule be saved)
│   │   │   └── Vintage Dreams Waves v2.sf2
│   │   └── Result
│   │       ├── Midi (A folder where generated output in .mid files are saved)
│   │       ├── npy (A folder where generated output in .npy files are saved)
│   │       ├── Performance (A folder where visualized metric plots are saved)
│   │       ├── Rolls (A folder where visualged images of generated outputs are saved)
│   │       └── wav (A folder where audio converted from generated outputs files are saved)
│   └── trained_model
│       └── musegan-new (A folder where model informations are saved)
│           ├── musegan.h5
│           ├── musegan_training_log.csv
│           └── musegan-checkpoints (A folder where model training checkpoints are saved)
│                ├── musegan_epoch_01.h5
│                ├── musegan_epoch_02.h5
│                ├── . . .
│                └── musegan_epoch_20.h5
</pre>

1. Save raw Lakh MIDI Dataset inside "simiplified-musegan/dataset/Raw/Lakh/".
2. Run "simiplified-musegan/Model/Capstone-Project/Training/preprocess_data.py".
   - Preprocessed data will be saved in "simiplified-musegan/dataset/Preprocessed/Lakh/MultiTrack/".
3. Run "simiplified-musegan/Model/Capstone-Project/Training/model.py" to train model.
   - Remove or rename "musegan-new" folder at "simplified-musegan/Model/trained_model/musegan-new" to something else if you want to train from scratch.
   - Performance Metric Scores based on AUC, Precision, Recall, Loss will be saved in "simiplified-musegan/Model/Capstone-Project/Result/Performance".
5. Run "simiplified-musegan/Model/Capstone-Project/Evaluation/generate_output.py" to generate output.
   - Output piano roll will be saved in .npy format in "simiplified-musegan/Model/Capstone-Project/Result/npy/"
   - Output piano roll will be saved in .mid format in "simiplified-musegan/Model/Capstone-Project/Result/Midi/"
   - Output piano roll will be saved in .png format in "simiplified-musegan/Model/Capstone-Project/Result/Rolls/"
6. Run "simiplified-musegan/Model/Capstone-Project/Evaluation/evaluation.py" to evaluate the training data and generated output based on the metrics designed by Dong et al. (2018).
   - EB (Empty Bar Rate), Used Pitch Classes (UPC), Qualified Notes (QN), Drum Patterns (DP), Tonal Distance (TD)
   - Calculation algorithm might be different to those designed by Dong et al. (2018).
7. Run "simiplified-musegan/Model/Capstone-Project/Evaluation/convert_to_audio.py" to convert generated .mid file to .wav audio.
   - Audio in .wav format will be saved in "simiplified-musegan/Model/Capstone-Project/Result/wav/"
