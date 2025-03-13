from pydub import AudioSegment
import os
from glob import glob

INPUT_DIR = "/Users/nigel/Documents/noisy_data"
OUTPUT_DIR = "/Users/nigel/Documents/converted_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_to_standard_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit PCM
    audio.export(output_path, format="wav")

for audio_file in glob(os.path.join(INPUT_DIR, "*.*")):
    filename = os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
    output_path = os.path.join(OUTPUT_DIR, filename)
    convert_to_standard_wav(audio_file, output_path)
    print(f"Converted {audio_file} → {output_path}")
