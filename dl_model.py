import torch
import torch.nn as nn
import os
from dataclasses import dataclass


class SpeechRNN(nn.Module):
    """LSTM-based speech recognition model."""

    def __init__(self, num_classes=30):
        super(SpeechRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.out_layer = nn.Linear(256, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.out_layer(out[:, -1, :])
        return self.softmax(x)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the LSTM model and audio processing."""
    model_path: str = os.getenv("MODEL_PATH", "models/speech_lstm_model.pth")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1024"))
    record_seconds: int = int(os.getenv("RECORD_SECONDS", "1"))
