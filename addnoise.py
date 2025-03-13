#!/usr/bin/env python3
import os
import random
import numpy as np
from pydub import AudioSegment
from glob import glob
from datasets import load_dataset, Audio
from huggingface_hub import login
import json

# Hugging Face authentication
login("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")

# Choose your target language (must be available in Common Voice)
language = "zh-HK"

'''
 Available: ['ab', 'af', 'am', 'ar', 'as', 'ast', 'az', 'ba', 'bas', 'be', 'bg', 'bn', 'br', 'ca', 'ckb',
  'cnh', 'cs', 'cv', 'cy', 'da', 'de', 'dv', 'dyu', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy-NL',
   'ga-IE', 'gl', 'gn', 'ha', 'he', 'hi', 'hsb', 'hu', 'hy-AM', 'ia', 'id', 'ig', 'is', 'it', 'ja', 'ka', 'kab', 
   'kk', 'kmr', 'ko', 'ky', 'lg', 'lij', 'lo', 'lt', 'ltg', 'lv', 'mdf', 'mhr', 'mk', 'ml', 'mn', 'mr', 'mrj', 'mt',
    'myv', 'nan-tw', 'ne-NP', 'nhi', 'nl', 'nn-NO', 'oc', 'or', 'os', 'pa-IN', 'pl', 'ps', 'pt', 'quy', 'rm-sursilv',
     'rm-vallader', 'ro', 'ru', 'rw', 'sah', 'sat', 'sc', 'sk', 'skr', 'sl', 'sq', 'sr', 'sv-SE', 'sw', 'ta', 'te',
      'th', 'ti', 'tig', 'tk', 'tok', 'tr', 'tt', 'tw', 'ug', 'uk', 'ur', 'uz', 'vi', 'vot', 'yi', 'yo', 'yue', 'zgh',
       'zh-CN', 'zh-HK', 'zh-TW']

'''

# Directory with noise audio files (make sure these are .wav files)
NOISE_DIR = r"C:\Users\nlk3212\Documents\Converted_noise"
# Output directory to save noisy audio and transcript files
OUTPUT_DIR = rf"C:\Users\nlk3212\Documents\noisyData\{language}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all noise files from NOISE_DIR
noise_files = glob(os.path.join(NOISE_DIR, "*.wav"))

def create_noise_with_intervals(target_duration_ms, noise_files,
                                min_noise_interval=3000, max_noise_interval=8000,
                                min_silence_interval=2000, max_silence_interval=5000,
                                noise_volume_db=-20):
    """
    Create a noise AudioSegment with intervals of noise and silence to match target duration.
    """
    combined_noise = AudioSegment.silent(duration=0)
    duration = 0

    while duration < target_duration_ms:
        noise_interval = random.randint(min_noise_interval, max_noise_interval)
        noise_audio = AudioSegment.from_file(random.choice(noise_files))

        if len(noise_audio) < noise_interval:
            repeats = (noise_interval // len(noise_audio)) + 1
            noise_audio *= repeats

        noise_audio = noise_audio[:noise_interval] - abs(noise_volume_db)
        combined_noise += noise_audio
        duration += noise_interval

        if duration >= target_duration_ms:
            break

        silence_interval = random.randint(min_silence_interval, max_silence_interval)
        silence_audio = AudioSegment.silent(duration=silence_interval)
        combined_noise += silence_audio
        duration += silence_interval

    return combined_noise[:target_duration_ms]

print(f"Processing language: {language}")
# Load the Common Voice dataset (streaming)
ds = load_dataset("mozilla-foundation/common_voice_16_1", language, split="train", streaming=True, trust_remote_code=True)
# Cast the audio column to a specific sampling rate (16 kHz)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# List to hold manifest entries
manifest_entries = []

for idx, example in enumerate(ds):
    try:
        # Get the raw audio and its sample rate
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        # Convert audio array (float32) to int16 for proper PCM 16-bit data
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Create an AudioSegment from the int16 bytes
        original_audio = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sampling_rate,
            sample_width=2,  # 16-bit PCM = 2 bytes
            channels=1
        )

        # Create a noise background that lasts as long as the original audio
        noisy_background = create_noise_with_intervals(len(original_audio), noise_files)
        # Overlay the noise on the original audio
        combined_audio = original_audio.overlay(noisy_background)

        # Build filenames for audio and transcript
        base_filename = f"noisy_{language}_{idx}"
        audio_filename = base_filename + ".wav"
        transcript_filename = base_filename + ".txt"

        audio_output_path = os.path.join(OUTPUT_DIR, audio_filename)
        transcript_output_path = os.path.join(OUTPUT_DIR, transcript_filename)

        # Export the combined noisy audio to a WAV file
        combined_audio.export(audio_output_path, format="wav", bitrate="16k")

        # Extract transcription (assuming it's stored under the "sentence" key)
        transcript = example.get("sentence", "")

        # Write the transcript to a .txt file (optional but needed for your preprocessing)
        with open(transcript_output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)

        # Calculate duration in seconds (pydub gives duration in ms)
        duration = len(combined_audio) / 1000.0

        # Add an entry for this sample to the manifest
        manifest_entries.append({
            "audio_filepath": audio_output_path,
            "duration": duration,
            "text": transcript
        })

        print(f"✅ Saved: {audio_filename} with transcript")

        # Limit to a certain number of examples (optional)
        if idx >= 100:
            break

    except Exception as e:
        print(f"❌ Error ({language}, {idx}): {e}")

# Save the manifest as a JSON Lines file in the output directory
manifest_path = os.path.join(OUTPUT_DIR, "manifest.jsonl")
with open(manifest_path, 'w', encoding='utf-8') as mf:
    for entry in manifest_entries:
        mf.write(json.dumps(entry) + "\n")

print(f"Manifest saved to {manifest_path}")
