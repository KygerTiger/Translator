from transformers import WhisperProcessor

# Load the original processor you used during training
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Save it to the same directory where your checkpoints live
processor.save_pretrained(r"C:\Users\nlk3212\Documents\TunedTests\whispertunedEX1")
