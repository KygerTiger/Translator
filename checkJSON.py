import json

def check_jsonl_manifest(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file, start=1):
            try:
                data = json.loads(line)
                if "audio_filepath" not in data or "text" not in data:
                    print(f"❌ Line {idx}: Missing required fields.")
                    print(f"  Contents: {line.strip()}")
                else:
                    print(f"✅ Line {idx}: Valid entry.")
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {idx}: Invalid JSON - {e}")
                print(f"  Contents: {line.strip()}")

# Replace with your manifest file path
manifest_file = '/Users/nigel/Documents/withNoise_preprocessed/preprocessed_manifest.json'
check_jsonl_manifest(manifest_file)

def validate_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ Error on line {idx}: {e}")
                return False
    print("✅ All lines are valid JSON!")
    return True

validate_jsonl(manifest_file)
