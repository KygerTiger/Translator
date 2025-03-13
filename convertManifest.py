import os
import json

# Paths to your old and new manifest files
old_manifest_path = "/Users/nigel/Documents/withNoise_preprocessed/preprocessed_manifest.json"  # Replace with your actual manifest path
new_manifest_path = "/Users/nigel/Documents/withNoise_preprocessed/corrected_manifest.json"  # Output manifest with updated paths

# The new base directory where your audio files are stored
new_base_dir = "/home/nkyger/whisper-finetune/"

with open(old_manifest_path, 'r', encoding='utf-8') as old_manifest, \
        open(new_manifest_path, 'w', encoding='utf-8') as new_manifest:
    for line in old_manifest:
        entry = json.loads(line)
        # Extract just the filename from the old path
        filename = os.path.basename(entry["audio_filepath"])
        # Update the path to point to the new directory
        entry["audio_filepath"] = os.path.join(new_base_dir, filename)
        # Write the updated entry to the new manifest file
        new_manifest.write(json.dumps(entry) + "\n")

print(f"Updated manifest saved to {new_manifest_path}")
