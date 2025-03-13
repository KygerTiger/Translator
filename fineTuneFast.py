#!/usr/bin/env python
import os
import argparse
import math
import numpy as np
import torch
import librosa
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
from tqdm.auto import tqdm
from huggingface_hub import HfApi, HfFolder
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# --- Set up generation configuration ---
gen_config = GenerationConfig(
    max_length=448,
    suppress_tokens=[1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362],
    begin_suppress_tokens=[220, 50257]
)

# ---------------------------
# Data Collator for Speech Seq2Seq Tasks
# ---------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    max_target_length: int = 448  # Maximum allowed length for label tokens

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Convert and pad audio features to fixed length 3000 time steps.
        input_features = [feature["input_features"] for feature in features]
        padded_audio = []
        for feat in input_features:
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat)
            current_length = feat.size(-1)
            if current_length < 3000:
                pad_amount = 3000 - current_length
                feat = F.pad(feat, (0, pad_amount), mode="constant", value=0)
            elif current_length > 3000:
                feat = feat[..., :3000]
            padded_audio.append(feat)
        padded_audio = torch.stack(padded_audio)

        # Process labels using the tokenizer's pad method.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding="longest", return_tensors="pt", return_attention_mask=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if labels.shape[1] > self.max_target_length:
            labels = labels[:, :self.max_target_length]
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        return {"input_features": padded_audio, "labels": labels}

# ---------------------------
# Custom progress callback with tqdm
# ---------------------------
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

# ---------------------------
# Main training function
# ---------------------------
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
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size")
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
    # Set the generation config's start token values using the tokenizer’s bos token
    gen_config.bos_token_id = processor.tokenizer.bos_token_id
    gen_config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.generation_config = gen_config

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")

    # 2. Load JSON Lines manifests as Datasets
    print("Loading datasets from JSON Lines...")
    train_dataset = load_dataset("json", data_files=args.train_manifest, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_manifest, split="train")

    # Rename "text" to "transcription" for consistency
    if "text" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("text", "transcription")
    if "text" in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column("text", "transcription")

    # Optionally, filter out entries with empty transcriptions if needed.
    train_dataset = train_dataset.filter(lambda x: x.get("transcription", "").strip() != "")
    eval_dataset = eval_dataset.filter(lambda x: x.get("transcription", "").strip() != "")

    print("Training dataset size after filtering:", len(train_dataset))
    print("Evaluation dataset size after filtering:", len(eval_dataset))
    # Use entire dataset for fine-tuning (no subset selection)

    # 3. Define the preprocessing function.
    def preprocess_function(examples):
        inputs = []
        for audio_file in examples["audio_filepath"]:
            try:
                speech_array, _ = librosa.load(audio_file, sr=16000)
            except Exception as e:
                print(f"Warning: Could not load {audio_file}. Error: {e}")
                speech_array = np.zeros(16000, dtype=np.float32)
            inputs.append(speech_array)

        processed = processor(
            inputs, sampling_rate=16000, return_tensors="pt",
            padding="max_length", max_length=3000
        )
        model_inputs = {"input_features": processed.input_features}

        try:
            tokenized = processor.tokenizer(
                examples["transcription"],
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
        except Exception as e:
            print("Error tokenizing transcription:", e)
            tokenized = {"input_ids": torch.zeros((len(examples["transcription"]), 20), dtype=torch.long)}
        model_inputs["labels"] = tokenized.input_ids
        return model_inputs

    # 4. Map the preprocessing function (using parallel processing)
    print("Preprocessing datasets...")
    original_train_columns = train_dataset.column_names
    original_eval_columns = eval_dataset.column_names

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_train_columns,
        num_proc=4
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_eval_columns,
        num_proc=4
    )

    # 5. Set up training arguments (using the full dataset)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=2,
    )

    # 6. Load evaluation metric (WER)
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

    # 8. Initialize a dedicated data collator using our custom class.
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.bos_token_id,
        max_target_length=448,
    )

    # 9. Initialize the Trainer using processing_class to avoid deprecated tokenizer param.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[ProgressPercentageCallback(total_steps=total_steps)],
        data_collator=data_collator,
    )

    # 10. Fine-tuning
    print("Starting fine tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    processor.save_pretrained(args.output_dir)
    # 11. Evaluate the fine-tuned model
    print("Evaluating the fine tuned model...")
    results = trainer.evaluate()
    print("Evaluation results:", results)
    '''
    # 12. Optionally test a single audio file
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

    # 13. Optionally push the fine-tuned model to the Hugging Face Hub
    if args.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        token = HfFolder.get_token()
        if token is None:
            print("No Hugging Face token found. Please run 'huggingface-cli login' first.")
        else:
            trainer.push_to_hub() '''

if __name__ == "__main__":
    main()
