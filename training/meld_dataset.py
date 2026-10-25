import os
import subprocess
import wave

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6,
        }

        self.sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}

    def _load_video_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Trye and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not open video: {video_path}")

            # reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error {e!s}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError("No frames could be extracted")

        # pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # before permute: [frames, height, width, channels]
        # after permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path: str):
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    audio_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Read using soundfile directly
            waveform_np, sample_rate = sf.read(audio_path)
            # Convert to tensor and add the channel dimension (1, frames) to match torchaudio's expected output shape
            waveform = torch.FloatTensor(waveform_np).unsqueeze(0)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
            )

            mel_spec = mel_spectogram(waveform)

            # normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            # pad and truncate
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction using ffmpeg failed: {e!s}")
        except Exception as e:
            raise ValueError(f"Audio error: {e!s}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]

        try:
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"

            video_path = os.path.join(self.video_dir, video_filename)
            video_path_exists = os.path.exists(video_path)

            if video_path_exists == False:
                raise FileNotFoundError(f"No video found for filename: {video_path}")

            text_inputs = self.tokenizer(
                row["Utterance"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            video_frames = self._load_video_frames(video_path)
            audio_features = self._extract_audio_features(video_path)

            # map emotion and sentiment labels
            emotion_label = self.emotion_map[row["Emotion"].lower()]
            sentiment_label = self.sentiment_map[row["Sentiment"].lower()]

            return {
                "text_inputs": {
                    "input_ids": text_inputs["input_ids"].squeeze(),
                    "attention_mask": text_inputs["attention_mask"].squeeze(),
                },
                "video_frames": video_frames,
                "audio_features": audio_features,
                "emotion_label": torch.tensor(emotion_label),
                "sentiment_label": torch.tensor(sentiment_label),
            }

        except Exception as e:
            print(f"Error processing {video_path}: {e!s}")
            return None


def collate_fn(batch):
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(
    train_csv,
    train_video_dir,
    dev_csv,
    dev_video_dir,
    test_csv,
    test_video_dir,
    batch_size=32,
):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        "../dataset/train/train_sent_emo.csv",
        "../dataset/train/train_splits",
        "../dataset/dev/dev_sent_emo.csv",
        "../dataset/dev/dev_splits_complete",
        "../dataset/test/test_sent_emo.csv",
        "../dataset/test/output_repeated_splits_test",
    )

    for batch in train_loader:
        print(batch["text_inputs"])
        print(batch["video_frames"].shape)
        print(batch["audio_features"].shape)
        print(batch["emotion_label"])
        print(batch["sentiment_label"])
        break
