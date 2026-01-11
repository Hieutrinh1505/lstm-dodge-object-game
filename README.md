# LSTM Voice-Controlled Dodge Game

A real-time voice-controlled pygame where you dodge falling objects by speaking commands. This project combines deep learning (LSTM-based speech recognition) with game development to create an interactive voice-controlled gaming experience.

---

## Project Overview

This project implements a complete pipeline from **speech recognition model training** to **real-time game control**:

1. **Training Phase** ([Lab3_Speech_To_Text_student.ipynb](Lab3_Speech_To_Text_student.ipynb)): Train an LSTM model on Google's Speech Commands Dataset
2. **Inference Phase** ([main.py](main.py)): Use the trained model to control a Pygame character in real-time

---

## Key Takeaways

### 1. Deep Learning Concepts

#### **LSTM (Long Short-Term Memory) Networks**
- **Why LSTM over regular RNN?** Standard RNNs suffer from vanishing gradient problems and short-term memory issues. LSTMs solve this with internal "gates" that regulate information flow.
- **Architecture**: 2-layer LSTM with 256 hidden units
  - Input: 12 MFCC features per time step
  - Output: 30 word classes (probabilities)
  - Dropout: 0.3 for regularization

#### **MFCC (Mel-Frequency Cepstral Coefficients)**
- **What**: Audio feature extraction technique that mimics human auditory perception
- **Why**: Represents sound the way humans hear it, making it ideal for speech recognition
- **Parameters**: 12 coefficients, log_mels=True
- **Shape**: Converts 1-second audio (16000 samples) → (12, 81) feature matrix
  - 12 = number of MFCC coefficients
  - 81 = number of time windows

#### **Training Results**
- Dataset: Google Speech Commands (65,000 one-second utterances, 30 words)
- Final Test Accuracy: **88.87%**
- Training progression (5 epochs):
  - Epoch 0: 60.6% → 80.0% (train → val)
  - Epoch 4: 92.0% → 89.1% (train → val)

---

### 2. Real-Time Audio Processing

#### **Two-Thread Streaming Architecture**
The game uses a non-blocking architecture to achieve real-time voice control:

```
┌─────────────────────┐         ┌──────────────────────┐
│ audio_stream_thread │────────→│   Rolling Buffer     │
│  (Fast, ~60 Hz)     │         │   (deque, 1 sec)     │
└─────────────────────┘         └──────────┬───────────┘
                                           │
                                           ↓
                                ┌──────────────────────┐
                                │ prediction_thread    │
                                │ (LSTM inference)     │
                                └──────────┬───────────┘
                                           │
                                           ↓
                                ┌──────────────────────┐
                                │  Game Loop (60 FPS)  │
                                │  handle_voice_control│
                                └──────────────────────┘
```

**Thread 1: Audio Streaming**
- Continuously reads small chunks (1024 samples) from microphone
- Non-blocking: completes in milliseconds
- Appends to rolling buffer (deque with max length)

**Thread 2: Voice Prediction**
- Processes buffered audio when full 1-second window is available
- Runs LSTM inference on MFCC features
- Returns top 3 predictions (not just top 1!)
- Updates game state through thread-safe locks

**Why this works**: Separates fast I/O (audio streaming) from slow computation (LSTM inference), preventing game loop blocking.

---

### 3. Game Design Patterns

#### **Top-3 Prediction Strategy**
Instead of requiring the model to be 100% confident, the game accepts commands if they appear in the top 3 predictions:

```python
# Check if "left" or "right" is in top 3 predictions
for i, cmd in enumerate(top3_commands):
    cmd_lower = cmd.lower()
    if cmd_lower in ["left", "right"]:
        detected_command = cmd_lower
        break
```

**Why?** Words like "right" might be harder to predict (similar to "write", "night", etc.), but if it's in the top 3, that's good enough for game control.

#### **Command Cooldown Mechanism**
- Cooldown: 200ms between command applications
- Prevents excessive movement (60 FPS × no cooldown = 60 moves/second)
- Provides smooth, controlled character movement

#### **Progressive Difficulty**
- Enemies spawn every 60 frames initially
- Spawn delay decreases over time (minimum 20 frames)
- Enemy speed: 2 pixels/frame
- Score: +10 points per dodged enemy

---

### 4. Core Technical Stack

#### **Audio Processing**
- **PyAudio**: Real-time microphone access
- **torchaudio**: MFCC transformation
- Sample rate: 16000 Hz
- Chunk size: 1024 samples
- Audio normalization: int16 → float32 ÷ 32768.0

#### **Deep Learning**
- **PyTorch**: Model inference (no training in game)
- Model format: `.pth` checkpoint with state dict
- Inference mode: `model.eval()` + `torch.no_grad()`

#### **Game Development**
- **Pygame**: Game loop, rendering, collision detection
- Frame rate: 60 FPS (locked with `clock.tick(60)`)
- Screen: 800×600 pixels

#### **Concurrency**
- **threading**: Background audio processing
- **deque**: Thread-safe rolling buffer
- **threading.Lock()**: Synchronize command updates

---

### 5. Project Structure

```
lstm-dodge-object-game/
│
├── Lab3_Speech_To_Text_student.ipynb  # Training notebook (CRITICAL)
│   ├── Dataset download & exploration
│   ├── MFCC feature extraction
│   ├── LSTM model definition
│   ├── Training loop (5 epochs)
│   └── Model export → models/speech_lstm_model.pth
│
├── main.py                             # Game + inference
│   ├── Model loading
│   ├── Real-time audio streaming
│   ├── Voice prediction threads
│   └── Pygame game loop
│
├── dl_model.py                         # Model architecture
│   ├── SpeechRNN (LSTM definition)
│   └── ModelConfig (environment-based config)
│
├── utils.py                            # Audio preprocessing
│   ├── mfcc_transform()
│   └── preprocess_audio()              # MUST match training!
│
├── models/
│   └── speech_lstm_model.pth          # Trained model checkpoint
│
├── requirements.txt                    # Python dependencies
└── .env.example                        # Configuration template
```

---

### 6. Critical Implementation Details

#### **MFCC Preprocessing Must Match Training**
The [utils.py](utils.py) preprocessing **MUST** be identical to training:

```python
# Training (Jupyter notebook)
mfcc = torchaudio.transforms.MFCC(n_mfcc=12, log_mels=True)(waveform)

# Inference (utils.py) - MUST BE IDENTICAL
mfcc_transformer = torchaudio.transforms.MFCC(n_mfcc=12, log_mels=True)
```

**Why?** Even slight differences (different n_mfcc, sample_rate, normalization) will cause model to fail.

#### **Audio Buffer Size Calculation**
```python
buffer_maxlen = sample_rate * record_seconds / chunk_size
              = 16000 * 1 / 1024
              = ~15.6 chunks
```

The buffer holds exactly 1 second of audio, continuously rolling.

#### **Thread Safety**
```python
# Two locks for different purposes
buffer_lock = threading.Lock()  # Protects audio_buffer (deque)
audio_lock = threading.Lock()   # Protects last_command (shared state)
```

---

### 7. Voice Commands Supported

**30 Word Classes** (from Google Speech Commands Dataset):
```
bed, bird, cat, dog, down, eight, five, four, go, happy,
house, left, marvin, nine, no, off, on, one, right, seven,
sheila, six, stop, three, tree, two, up, wow, yes, zero
```

**Game Control Mapping**:
- **"left"** → Move player left (speed × 1.5)
- **"right"** → Move player right (speed × 1.5)
- **"six"** → Also moves right (user-defined alias)

You can easily extend this by modifying [main.py:168-171](main.py#L168-L171).

---

### 8. How to Run

#### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **Step 2: Configure Environment**
```bash
cp .env.example .env
# Edit .env if needed (default values work fine)
```

#### **Step 3: Train Model** (Optional - model already included)
```bash
jupyter notebook Lab3_Speech_To_Text_student.ipynb
# Run all cells to train LSTM model
# Model saved to: models/speech_lstm_model.pth
```

#### **Step 4: Run Game**
```bash
python main.py
```

**Controls**:
- Speak "left" or "right" to move
- Arrow keys (keyboard fallback)
- ESC to quit
- R to restart after game over

---

### 9. Performance Considerations

#### **Latency Breakdown**
1. Audio chunk read: ~16ms (1024 samples @ 16kHz)
2. MFCC computation: ~5-10ms
3. LSTM inference: ~20-50ms (CPU)
4. Total: ~50-75ms latency (acceptable for gaming)

#### **CPU Usage**
- Game loop: ~5-10% (single core)
- Audio streaming: ~2-3%
- LSTM inference: ~10-15% (bursts)
- Total: ~20-30% CPU usage

#### **Potential Optimizations**
- Use GPU for inference (change `map_location="cpu"` to `"cuda"`)
- Reduce LSTM layers or hidden size for faster inference
- Increase chunk size (reduces I/O overhead, increases latency)

---

### 10. Common Challenges & Solutions

#### **Challenge 1: "Right" is hard to predict**
**Solution**: Use top-3 predictions instead of top-1. If "right" is anywhere in top 3, accept it.

#### **Challenge 2: Character moves too much**
**Solution**: Add command cooldown (200ms). Only apply commands if sufficient time has passed.

#### **Challenge 3: Game freezes during voice input**
**Solution**: Use two-thread architecture. Never block the game loop for audio processing.

#### **Challenge 4: Model accuracy drops in real-time**
**Solution**: Ensure preprocessing exactly matches training. Check MFCC parameters, sample rate, normalization.

---

### 11. Learning Outcomes

By completing this project, you learn:

✅ **Deep Learning**: LSTM architecture, sequence modeling, audio classification
✅ **Audio Processing**: MFCC features, real-time streaming, buffering strategies
✅ **Game Development**: Pygame basics, collision detection, game loops
✅ **Concurrent Programming**: Threading, locks, non-blocking I/O
✅ **Model Deployment**: Moving from training (Jupyter) to production (Python script)
✅ **System Integration**: Combining ML models with real-time applications

---

### 12. Extension Ideas

1. **Multi-word Commands**: "move left fast", "stop", "jump"
2. **Difficulty Levels**: Easy/Medium/Hard with different spawn rates
3. **Power-ups**: "shield", "slow time" activated by voice
4. **Multiplayer**: Two players, each controlling their own character
5. **Custom Vocabulary**: Train model on your own voice commands
6. **Mobile Deployment**: Port to mobile using Kivy or PyTorch Mobile

---

## References

- [Google Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
- [LSTM Original Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Pygame Documentation](https://www.pygame.org/docs/)

---

## License

This project is for educational purposes. Dataset from Google (Creative Commons).

---

**Built with ❤️ using PyTorch, Pygame, and Coffee**
