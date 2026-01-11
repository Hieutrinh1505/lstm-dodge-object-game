import torch
import numpy as np
import torchaudio
from typing import Callable


def mfcc_transform(sample_rate: int = 16000) -> torchaudio.transforms.MFCC:
    """Create MFCC transform with specified sample rate."""
    return torchaudio.transforms.MFCC(n_mfcc=12, log_mels=True, sample_rate=sample_rate)


def preprocess_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    mfcc_transformer: Callable,
) -> torch.Tensor:
    """Convert raw audio to MFCC features for model inference."""
    # Convert to float32
    audio_data = np.asarray(audio_data, dtype=np.float32)

    if audio_data.size == 0:
        raise ValueError("Audio data is empty.")

    # Convert to tensor
    waveform = torch.from_numpy(audio_data)

    # Ensure waveform is 2D: (channels, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim > 2:
        raise ValueError("Audio data must be 1D or 2D.")

    # Clamp to [-1, 1] range
    waveform = torch.clamp(waveform, -1.0, 1.0)

    # Target length (1 second)
    target_length = int(sample_rate * 1.0)
    if waveform.shape[1] < target_length:
        # Pad with zeros
        waveform = torch.nn.functional.pad(
            waveform,
            (0, target_length - waveform.shape[1]),
            mode="constant",
            value=0,
        )
    else:
        waveform = waveform[:, :target_length]

    # Compute MFCC
    mfcc = mfcc_transformer(waveform)
    mfcc = mfcc.squeeze(0).transpose(0, 1).unsqueeze(0)  # (1, 81, 12)

    return mfcc
