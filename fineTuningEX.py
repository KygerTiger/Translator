#!/usr/bin/env python
import os
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch
import librosa
import math
import numpy as np
from transformers import DataCollatorForSeq2Seq
import json
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from tqdm.auto import tqdm
from huggingface_hub import HfApi, HfFolder

class ProgressPercentageCallback(TrainerCallback):
    """
    A custom callback to display a progress bar based on total training steps.
    """
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.pbar = tqdm(total=total_steps, desc="Training Progress", leave=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.current_step += 1
        self.pbar.update(1)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()
        return control

def main():
    parser = argparse.ArgumentParser(description="Fine tune Whisper model with JSON Lines manifest")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-base",
                        help="Pre-trained Whisper model identifier or path")
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to training dataset JSON Lines file with 'audio_filepath' and 'text' fields")
    parser.add_argument("--eval_manifest", type=str, required=True,
                        help="Path to evaluation dataset JSON Lines file with same format")
    parser.add_argument("--output_dir", type=str, default="./whisper_finetuned",
                        help="Directory to save the fine tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Eval batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--test_audio", type=str, default=None,
                        help="Path to an audio file for transcription testing after fine tuning")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="If set, push the fine-tuned model to the Hugging Face Hub at the end.")
    args = parser.parse_args()

    # 1. Load pre-trained model and processor
    print("Loading model and processor...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)

    # 2. Load JSON Lines manifests as Datasets
    #    We'll use "audio_filepath" and "text" as columns, then rename "text" -> "transcription".
    print("Loading datasets from JSON Lines...")

    # The 'json' loading script can read JSON *lines* if each line is a valid JSON object.
    # We'll specify `field='data'` only if needed, but typically "split='train'" is enough.
    train_dataset = load_dataset("json", data_files=args.train_manifest, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_manifest, split="train")

    # Rename "text" column to "transcription" so we can keep the same variable names as before
    if "text" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("text", "transcription")
    if "text" in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column("text", "transcription")

    #train_dataset = train_dataset.select(range(10))
    #eval_dataset = eval_dataset.select(range(10))

    # 3. Define the preprocessing function
    def preprocess_function(examples):
        # Debug prints: inspect the keys and a sample
        print("Batch keys:", list(examples.keys()))
        if "audio_filepath" in examples:
            print("Sample audio_filepath:", examples["audio_filepath"][:3])
        if "transcription" in examples:
            print("Sample transcription:", examples["transcription"][:3])

        # Load audio files, with fallback to 1 second of silence on error
        inputs = []
        for audio_file in examples["audio_filepath"]:
            try:
                speech_array, _ = librosa.load(audio_file, sr=16000)
            except Exception as e:
                print(f"Warning: Could not load {audio_file}. Error: {e}")
                speech_array = np.zeros(16000, dtype=np.float32)
            inputs.append(speech_array)

        # Let the processor handle padding automatically
        processed = processor(inputs, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=3000)
        print("Processed input features shape:", processed.input_features.shape)

        # Build model inputs dictionary
        model_inputs = {"input_features": processed.input_features}

        # Tokenize transcriptions with exception handling
        try:
            tokenized = processor.tokenizer(
                examples["transcription"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            print("Error tokenizing transcription:", e)
            import torch
            tokenized = {"input_ids": torch.zeros((len(examples["transcription"]), 20), dtype=torch.long)}

        model_inputs["labels"] = tokenized.input_ids

        print(f"Processed {len(inputs)} audio files for a batch.")
        return model_inputs

    # 4. Map the preprocessing function
    print("Preprocessing datasets...")
    original_train_columns = train_dataset.column_names
    original_eval_columns = eval_dataset.column_names

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_train_columns,
        num_proc=4  # Use 4 processes (adjust based on your CPU cores)
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_eval_columns
    )


    #4.5 initialize custom data collector
    def custom_data_collator(features):
        # Process input_features: each should be a tensor of shape (80, seq_len)
        processed_features = []
        for f in features:
            feat = f["input_features"]
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat)
            processed_features.append(feat)

        # Compute the maximum sequence length along dimension 1
        max_seq_len = max(feat.shape[1] for feat in processed_features)

        padded_input_features = []
        for feat in processed_features:
            seq_len = feat.shape[1]
            pad_amount = max_seq_len - seq_len
            if pad_amount > 0:
                # Create padding of zeros along the time dimension (dim=1)
                pad_tensor = torch.zeros(feat.shape[0], pad_amount, dtype=feat.dtype)
                feat = torch.cat([feat, pad_tensor], dim=1)
            padded_input_features.append(feat)

        # Stack padded audio features into a single tensor (batch_size, 80, max_seq_len)
        padded_input_features = torch.stack(padded_input_features, dim=0)

        # Process labels: Convert lists to tensors if needed and pad them
        label_tensors = []
        for f in features:
            lab = f["labels"]
            if not isinstance(lab, torch.Tensor):
                lab = torch.tensor(lab)
            label_tensors.append(lab)

        # Use pad_sequence for 1D label tensors; pad with -100 (common ignore index)
        padded_labels = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100)

        return {"input_features": padded_input_features, "labels": padded_labels}


    # 5. Set up training arguments

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=10,  # Run only 10 steps for quick testing
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
    )

    # 6. Load evaluation metric (Word Error Rate)
    metric = load_metric("wer")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode predictions
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        # Clean labels (remove -100 tokens)
        labels = [[token for token in label if token != -100] for label in labels]
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        wer_score = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"wer": wer_score}

    # 7. Calculate total training steps (approx) for the progress bar
    steps_per_epoch = math.ceil(len(train_dataset) / training_args.per_device_train_batch_size)
    total_steps = steps_per_epoch * training_args.num_train_epochs

     #NEW: Create a dedicated Data Collator for Seq2Seq that handles padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True  # Ensures that inputs and labels are padded to uniform lengths
    )


    # 8. Initialize Trainer with custom progress callback
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,  # used for generation/decoding
        compute_metrics=compute_metrics,
        callbacks=[ProgressPercentageCallback(total_steps=total_steps)],
        data_collator=data_collator,  # use the custom collator
    )

    # 9. Fine-tuning
    print("Starting fine tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # 10. Evaluate on the eval dataset
    print("Evaluating the fine tuned model...")
    results = trainer.evaluate()
    print("Evaluation results:", results)

    # 11. Optionally test a single audio file
    if args.test_audio:
        print(f"Transcribing test audio: {args.test_audio}")
        speech_array, _ = librosa.load(args.test_audio, sr=16000)
        input_features = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features
        generated_ids = model.generate(input_features, max_new_tokens=448)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Test audio transcription:", transcription)
        # Save transcription to a file
        test_outfile = os.path.join(args.output_dir, "test_audio_transcription.txt")
        with open(test_outfile, "w") as f:
            f.write(transcription)
        print(f"Transcription saved to {test_outfile}")

    # 12. Optionally push the fine-tuned model to the Hugging Face Hub
    if args.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        # Ensure you have 'huggingface_hub' installed and have run 'huggingface-cli login'
        token = HfFolder.get_token()
        if token is None:
            print("No Hugging Face token found. Please run 'huggingface-cli login' first.")
        else:
            trainer.push_to_hub()

if __name__ == "__main__":
    main()
