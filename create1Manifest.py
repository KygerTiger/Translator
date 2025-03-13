#!/usr/bin/env python3
import os
import json
import argparse
import librosa


def gather_files_recursively(root_dir, exts=('.wav', '.mp3', '.flac', '.ogg'), compute_duration=False):
    """
    Recursively traverse 'root_dir' for audio files with the given extensions.
    For each audio file, look for a matching .txt transcript.
    Optionally compute duration (requires librosa).
    Returns a list of dicts with keys: audio_filepath, text, and optionally duration.
    """
    manifest_entries = []

    for current_path, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(exts):
                audio_path = os.path.join(current_path, file_name)

                # Match transcript by swapping extension for .txt
                base = os.path.splitext(audio_path)[0]
                transcript_path = base + '.txt'
                transcript = ""

                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()

                entry = {
                    "audio_filepath": audio_path,
                    "text": transcript
                }

                # If you also want to store duration in the manifest, compute it here
                if compute_duration:
                    audio_data, sr = librosa.load(audio_path, sr=None)
                    duration = librosa.get_duration(y=audio_data, sr=sr)
                    entry["duration"] = duration

                manifest_entries.append(entry)

    return manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Create a single manifest from a directory with subfolders.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Top-level directory containing subfolders of noisy audio + transcripts.")
    parser.add_argument("--manifest_path", type=str, required=True,
                        help="Path to save the output manifest file (JSON Lines format).")
    parser.add_argument("--compute_duration", action="store_true",
                        help="If set, compute the duration of each audio file using librosa.")
    args = parser.parse_args()

    manifest_entries = gather_files_recursively(
        root_dir=args.input_dir,
        compute_duration=args.compute_duration
    )

    # Write the combined manifest
    with open(args.manifest_path, 'w', encoding='utf-8') as mf:
        for entry in manifest_entries:
            mf.write(json.dumps(entry) + "\n")

    print(f"Manifest with {len(manifest_entries)} entries saved to {args.manifest_path}")


if __name__ == "__main__":
    main()
