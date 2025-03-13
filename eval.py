import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from evaluate import load
import json

# Set your paths
checkpoint_root = r"C:\Users\nlk3212\Documents\TunedTests\whispertunedEX2"
audio_test_file = r"C:\Users\nlk3212\Documents\Sound\best.wav"
ground_truth_transcript = "Hi how are you, What's your name? Wonderful weather we're having today isn't it?"  # Change this

# Load WER metric
wer_metric = load("wer")

# Collect checkpoints
checkpoints = sorted([
    os.path.join(checkpoint_root, d) for d in os.listdir(checkpoint_root)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_root, d))
], key=lambda x: int(x.split("-")[-1]))

results = []

for ckpt in checkpoints:
    print(f"Evaluating {os.path.basename(ckpt)}...")
    try:
        processor = WhisperProcessor.from_pretrained(checkpoint_root)
        model = WhisperForConditionalGeneration.from_pretrained(ckpt).eval()

        if torch.cuda.is_available():
            model.to("cuda")

        # Load and process test audio
        speech_array, _ = librosa.load(audio_test_file, sr=16000)
        inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"])
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        wer = wer_metric.compute(predictions=[transcription], references=[ground_truth_transcript])
        results.append((os.path.basename(ckpt), wer, transcription))

    except Exception as e:
        print(f"❌ Error evaluating {ckpt}: {e}")
        results.append((os.path.basename(ckpt), None, f"Error: {e}"))

# Rank by WER
print("\n📊 Evaluation Results:")
results = sorted(results, key=lambda x: float("inf") if x[1] is None else x[1])
for ckpt, wer, trans in results:
    print(f"Checkpoint: {ckpt} | WER: {wer if wer is not None else 'Error'}")
    print(f"Transcript: {trans}\n")

# Show training args if available
args_path = os.path.join(checkpoint_root, "training_args.bin")
config_path = os.path.join(checkpoint_root, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
        print("\n🔧 Model Configuration:")
        for k, v in config.items():
            print(f"{k}: {v}")
