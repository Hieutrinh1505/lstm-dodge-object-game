import numpy as np
import torch
import pygame
import random
import pyaudio
import threading
from collections import deque
from dl_model import SpeechRNN, ModelConfig
from utils import mfcc_transform, preprocess_audio

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("LSTM Dodge Game - Voice Controlled")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)



# Player
player = pygame.Rect(SCREEN_WIDTH // 2 - 25, SCREEN_HEIGHT - 80, 50, 50)
player_speed = 7

# Enemies (falling objects)
enemies = []
enemy_speed = 2
spawn_timer = 0
spawn_delay = 60  # Spawn new enemy every 60 frames (1 second)

# Game state
score = 0
game_over = False

# Font
font = pygame.font.Font(None, 36)
big_font = pygame.font.Font(None, 72)

# Load model
MODEL_PATH = ModelConfig.model_path
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
classes = checkpoint["classes"]

# Initialize model
model = SpeechRNN(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Initialize MFCC transform
mfcc_transformer = mfcc_transform(sample_rate=ModelConfig.sample_rate)

# Audio settings
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=ModelConfig.sample_rate,
    input=True,
    frames_per_buffer=ModelConfig.chunk_size,
)

# Voice control state
last_command = None
last_command_time = 0
voice_control_active = True
audio_lock = threading.Lock()
command_cooldown = 0.2  # Seconds between voice command applications

# Audio buffer for streaming
audio_buffer = deque(
    maxlen=int(
        ModelConfig.sample_rate * ModelConfig.record_seconds / ModelConfig.chunk_size
    )
)
buffer_lock = threading.Lock()


def predict(audio_data, sample_rate=16000):
    """Predict spoken word from audio data."""
    mfcc = preprocess_audio(audio_data, sample_rate, mfcc_transformer)

    with torch.no_grad():
        output = model(mfcc)
        probs = torch.exp(output)
        confidence, idx = probs.max(1)

    return classes[idx.item()], confidence.item()


def spawn_enemy():
    """Create a new enemy at random x position at top of screen"""
    enemy_width = 30
    enemy_height = 30
    x = random.randint(0, SCREEN_WIDTH - enemy_width)
    y = -enemy_height  # Start above screen
    return pygame.Rect(x, y, enemy_width, enemy_height)


def reset_game():
    """Reset game to initial state"""
    global player, enemies, score, game_over, spawn_timer, spawn_delay
    player.x = SCREEN_WIDTH // 2 - 25
    player.y = SCREEN_HEIGHT - 80
    enemies = []
    score = 0
    game_over = False
    spawn_timer = 0
    spawn_delay = 60


def audio_stream_thread():
    """Background thread for continuous audio streaming into buffer."""
    global voice_control_active

    while voice_control_active:
        try:
            # Read small audio chunk (non-blocking, very fast)
            data = stream.read(ModelConfig.chunk_size, exception_on_overflow=False)

            with buffer_lock:
                audio_buffer.append(data)

        except Exception as e:
            print(f"Audio stream error: {e}")


def voice_prediction_thread():
    """Background thread for processing buffered audio and making predictions."""
    global last_command, voice_control_active

    while voice_control_active:
        try:
            # Check if we have enough data
            with buffer_lock:
                if len(audio_buffer) < audio_buffer.maxlen:
                    continue
                # Get all buffered audio
                buffered_frames = list(audio_buffer)

            # Convert to numpy array
            audio_data = np.frombuffer(b"".join(buffered_frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Predict command
            command, confidence = predict(audio_data, ModelConfig.sample_rate)

            # Update last command if confident
            if confidence > 0.6:  # Slightly higher threshold for better accuracy
                with audio_lock:
                    last_command = command.lower()
                    print(f"Detected: {command} (confidence: {confidence:.2f})")
            else:
                with audio_lock:
                    last_command = None

        except Exception as e:
            print(f"Voice prediction error: {e}")


def handle_voice_control():
    """Apply voice commands to player movement."""
    global last_command, last_command_time

    current_time = pygame.time.get_ticks() / 1000.0

    # Only apply command if cooldown has passed
    if current_time - last_command_time < command_cooldown:
        return

    with audio_lock:
        command = last_command

    if command:
        if command in ["left"]:  # Adjust based on your trained classes
            player.x -= player_speed * 1.5
        elif command in ["right","six"]:  # Adjust based on your trained classes
            player.x += player_speed * 1.5

        last_command_time = current_time


# Start voice control threads
audio_thread = threading.Thread(target=audio_stream_thread, daemon=True)
prediction_thread = threading.Thread(target=voice_prediction_thread, daemon=True)
audio_thread.start()
prediction_thread.start()

# Main game loop
running = True
while running:
    # ============================================
    # EVENT HANDLING
    # ============================================
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r and game_over:
                reset_game()

    if not game_over:
        # ============================================
        # UPDATE - Voice Control
        # ============================================
        handle_voice_control()

        # ============================================
        # UPDATE - Player Movement (Keyboard fallback)
        # ============================================
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            player.x -= player_speed
        if keys[pygame.K_RIGHT]:
            player.x += player_speed

        # Boundary check
        if player.left < 0:
            player.left = 0
        if player.right > SCREEN_WIDTH:
            player.right = SCREEN_WIDTH

        # ============================================
        # UPDATE - Spawn Enemies
        # ============================================
        spawn_timer += 1
        if spawn_timer >= spawn_delay:
            enemies.append(spawn_enemy())
            spawn_timer = 0

            # Increase difficulty over time
            if spawn_delay > 20:
                spawn_delay -= 1

        # ============================================
        # UPDATE - Move Enemies
        # ============================================
        for enemy in enemies[:]:  # [:] creates a copy to safely modify list
            enemy.y += enemy_speed

            # Remove if off screen (and add score)
            if enemy.top > SCREEN_HEIGHT:
                enemies.remove(enemy)
                score += 10

        # ============================================
        # UPDATE - Collision Detection
        # ============================================
        for enemy in enemies:
            # colliderect() returns True if rectangles overlap
            if player.colliderect(enemy):
                game_over = True
                break

    # ============================================
    # DRAWING
    # ============================================
    screen.fill(BLACK)

    # Draw player
    pygame.draw.rect(screen, BLUE, player)

    # Draw enemies
    for enemy in enemies:
        pygame.draw.rect(screen, RED, enemy)

    # Draw score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Draw enemy count
    enemy_text = font.render(f"Enemies: {len(enemies)}", True, WHITE)
    screen.blit(enemy_text, (10, 50))

    # Game over screen
    if game_over:
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        # Game over text
        game_over_text = big_font.render("GAME OVER", True, RED)
        text_rect = game_over_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
        )
        screen.blit(game_over_text, text_rect)

        final_score = font.render(f"Final Score: {score}", True, WHITE)
        score_rect = final_score.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20)
        )
        screen.blit(final_score, score_rect)

        restart_text = font.render("Press R to Restart", True, GREEN)
        restart_rect = restart_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60)
        )
        screen.blit(restart_text, restart_rect)

    # Instructions
    if not game_over:
        inst = font.render(
            "Voice or Arrow Keys to Move | Dodge the red squares!", True, WHITE
        )
        screen.blit(inst, (10, SCREEN_HEIGHT - 40))

    pygame.display.flip()
    clock.tick(60)

# Cleanup
voice_control_active = False
audio_thread.join(timeout=1)
prediction_thread.join(timeout=1)
stream.stop_stream()
stream.close()
audio.terminate()
pygame.quit()
