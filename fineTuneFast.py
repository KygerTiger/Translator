#!/usr/bin/env python
import os
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
import librosa
import math
import numpy as np
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


# Custom progress callback with tqdm
class ProgressPercentageCallback(TrainerCallback):
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
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Train batch size")  # increased default
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Eval batch size")  # increased default
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

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")

    # 2. Load JSON Lines manifests as Datasets
    print("Loading datasets from JSON Lines...")
    train_dataset = load_dataset("json", data_files=args.train_manifest, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_manifest, split="train")

    # Rename "text" to "transcription"
    if "text" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("text", "transcription")
    if "text" in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column("text", "transcription")

    # For quick testing, select a small subset (remove or adjust for full run)
    train_dataset = train_dataset.select(range(10))
    eval_dataset = eval_dataset.select(range(10))

    # 3. Define the preprocessing function (consider removing debug prints in production)
    def preprocess_function(examples):
        # (Optional) Debug prints can be commented out after testing:
        # print("Batch keys:", list(examples.keys()))
        # print("Sample audio_filepath:", examples["audio_filepath"][:3])
        # print("Sample transcription:", examples["transcription"][:3])

        inputs = []
        for audio_file in examples["audio_filepath"]:
            try:
                speech_array, _ = librosa.load(audio_file, sr=16000)
            except Exception as e:
                print(f"Warning: Could not load {audio_file}. Error: {e}")
                speech_array = np.zeros(16000, dtype=np.float32)
            inputs.append(speech_array)

        # Let the processor handle padding automatically
        processed = processor(inputs, sampling_rate=16000, return_tensors="pt", padding=True)
        # print("Processed input features shape:", processed.input_features.shape)

        model_inputs = {"input_features": processed.input_features}

        try:
            tokenized = processor.tokenizer(
                examples["transcription"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            print("Error tokenizing transcription:", e)
            tokenized = {"input_ids": torch.zeros((len(examples["transcription"]), 20), dtype=torch.long)}

        model_inputs["labels"] = tokenized.input_ids
        # print(f"Processed {len(inputs)} audio files for a batch.")
        return model_inputs

    # 4. Map the preprocessing function (using parallel processing)
    print("Preprocessing datasets...")
    original_train_columns = train_dataset.column_names
    original_eval_columns = eval_dataset.column_names

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_train_columns,
        num_proc=4  # Increase based on available CPU cores
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_eval_columns,
        num_proc=4
    )

    # 4.5. Define custom data collator for efficient padding of audio features and labels
    def custom_data_collator(features):
        processed_features = []
        for f in features:
            feat = f["input_features"]
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat)
            processed_features.append(feat)

        # Pad along the time dimension (assumes shape (80, seq_len))
        max_seq_len = max(feat.shape[1] for feat in processed_features)
        padded_input_features = []
        for feat in processed_features:
            seq_len = feat.shape[1]
            pad_amount = max_seq_len - seq_len
            if pad_amount > 0:
                pad_tensor = torch.zeros(feat.shape[0], pad_amount, dtype=feat.dtype)
                feat = torch.cat([feat, pad_tensor], dim=1)
            padded_input_features.append(feat)
        padded_input_features = torch.stack(padded_input_features, dim=0)

        label_tensors = []
        for f in features:
            lab = f["labels"]
            if not isinstance(lab, torch.Tensor):
                lab = torch.tensor(lab)
            label_tensors.append(lab)
        padded_labels = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        return {"input_features": padded_input_features, "labels": padded_labels}

    # 5. Set up training arguments (using max_steps for a quick run; remove for full training)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=100,  # For quick testing; remove or adjust for full training
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        dataloader_pin_memory=True,  # Enables faster host-to-device transfers
    )

    # 6. Load evaluation metric (Word Error Rate)
    metric = load_metric("wer")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        labels = [[token for token in label if token != -100] for label in labels]
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        wer_score = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"wer": wer_score}

    # 7. Calculate total training steps for progress callback
    steps_per_epoch = math.ceil(len(train_dataset) / training_args.per_device_train_batch_size)
    total_steps = steps_per_epoch * training_args.num_train_epochs

    # 8. Initialize Trainer with custom data collator and progress callback
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,  # used for generation/decoding
        compute_metrics=compute_metrics,
        callbacks=[ProgressPercentageCallback(total_steps=total_steps)],
        data_collator=custom_data_collator,
    )

    # 9. Fine-tuning
    print("Starting fine tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # 10. Evaluate the fine-tuned model
    print("Evaluating the fine tuned model...")
    results = trainer.evaluate()
    print("Evaluation results:", results)

    # 11. Optionally test a single audio file
    if args.test_audio:
        print(f"Transcribing test audio: {args.test_audio}")
        speech_array, _ = librosa.load(args.test_audio, sr=16000)
        input_features = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
        generated_ids = model.generate(input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Test audio transcription:", transcription)
        test_outfile = os.path.join(args.output_dir, "test_audio_transcription.txt")
        with open(test_outfile, "w") as f:
            f.write(transcription)
        print(f"Transcription saved to {test_outfile}")

    # 12. Optionally push the model to the Hugging Face Hub
    if args.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        token = HfFolder.get_token()
        if token is None:
            print("No Hugging Face token found. Please run 'huggingface-cli login' first.")
        else:
            trainer.push_to_hub()


if __name__ == "__main__":
    main()
