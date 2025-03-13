#!/usr/bin/env python3
import os
import argparse
import librosa
import soundfile as sf
import json

def preprocess_audio(file_path, target_sr=16000):
    """
    Load and preprocess an audio file.
    - Loads the file without resampling.
    - Resamples the audio to the target sample rate if needed.
    - Normalizes the audio to the range [-1, 1].
    """
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    max_val = max(abs(audio.max()), abs(audio.min()))
    if max_val > 0:
        audio = audio / max_val
    return audio, target_sr

def process_manifest(manifest_in, output_dir, manifest_out, target_sr=16000):
    """
    Process all audio files listed in the input manifest file.
    For each file:
      - Preprocess the audio.
      - Save the processed file to output_dir.
      - Compute its duration.
      - Retain the transcript (if available) from the manifest;
        if missing, try to load it from a .txt file with the same basename.
    Then, write a new manifest (JSON Lines file) with the updated information.
    """
    os.makedirs(output_dir, exist_ok=True)
    new_manifest = []

    with open(manifest_in, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            input_file = entry["audio_filepath"]
            transcript = entry.get("text", "").strip()

            # If transcript is empty, try to load from a corresponding .txt file.
            if not transcript:
                base = os.path.splitext(input_file)[0]
                transcript_path = base + '.txt'
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8') as tf:
                        transcript = tf.read().strip()

            # Preprocess the audio file.
            audio, sr = preprocess_audio(input_file, target_sr)

            # Save the processed audio file (same basename, new output directory).
            output_file = os.path.join(output_dir, os.path.basename(input_file))
            sf.write(output_file, audio, sr)

            # Compute duration (in seconds).
            duration = librosa.get_duration(y=audio, sr=sr)

            # Build the new manifest entry.
            new_manifest.append({
                "audio_filepath": output_file,
                "duration": duration,
                "text": transcript
            })
            print(f"Processed {os.path.basename(input_file)}: duration = {duration:.2f} s, transcript length = {len(transcript)}")

    # Save the new manifest as a JSON Lines file.
    with open(manifest_out, 'w', encoding='utf-8') as mf:
        for entry in new_manifest:
            mf.write(json.dumps(entry) + "\n")
    print(f"Manifest with {len(new_manifest)} entries saved to {manifest_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess audio data using an existing manifest for Whisper fine tuning")
    parser.add_argument("--manifest_in", type=str, required=True,
                        help="Path to the input manifest file (JSON Lines format)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save preprocessed audio files")
    parser.add_argument("--manifest_out", type=str, default="preprocessed_manifest.json",
                        help="Path to save the new manifest file (JSON Lines format)")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Target sample rate for audio (default: 16000 Hz)")
    args = parser.parse_args()

    process_manifest(args.manifest_in, args.output_dir, args.manifest_out, args.target_sr)
