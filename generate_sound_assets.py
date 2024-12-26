#!/usr/bin/env python3
"""
/src/audio/generate_tetris_sounds.py

A script to generate sound effects for a Tetris game using synthesized audio.
Generates WAV files for various game actions: move, rotate, clear, hold, drop, and game over.

Dependencies:
    - numpy
    - scipy

Install dependencies:
    pip install numpy scipy
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import os


class TetrisSoundGenerator:
    """Generate and save sound effects for Tetris game actions."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the sound generator.
        
        Args:
            sample_rate: Number of samples per second (default: 44100 Hz)
        """
        self.sample_rate = sample_rate
        self.output_dir = "sounds"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_sine_wave(self, freq: float, duration: float, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a sine wave.
        
        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            amplitude: Wave amplitude (default: 1.0)
            
        Returns:
            numpy.ndarray: Audio samples
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        return amplitude * np.sin(2 * np.pi * freq * t)

    def apply_envelope(self, audio: np.ndarray, attack: float = 0.01, decay: float = 0.1) -> np.ndarray:
        """
        Apply an ADSR envelope to the audio.
        
        Args:
            audio: Input audio samples
            attack: Attack time in seconds
            decay: Decay time in seconds
            
        Returns:
            numpy.ndarray: Processed audio samples
        """
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        
        envelope = np.ones_like(audio)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return audio * envelope

    def save_wav(self, audio: np.ndarray, filename: str):
        """
        Save audio data as a WAV file.
        
        Args:
            audio: Audio samples to save
            filename: Output filename
        """
        # Normalize audio to prevent clipping
        audio = audio / np.max(np.abs(audio))
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16)
        
        filepath = os.path.join(self.output_dir, filename)
        wavfile.write(filepath, self.sample_rate, audio_16bit)
        print(f"Generated: {filepath}")

    def generate_move_sound(self):
        """Generate sound for piece movement."""
        duration = 0.05
        freq = 200
        audio = self.generate_sine_wave(freq, duration)
        audio = self.apply_envelope(audio, attack=0.01, decay=0.02)
        self.save_wav(audio, "move.wav")

    def generate_rotate_sound(self):
        """Generate sound for piece rotation."""
        duration = 0.08
        freq = 300
        audio = self.generate_sine_wave(freq, duration)
        audio = self.apply_envelope(audio, attack=0.01, decay=0.03)
        self.save_wav(audio, "rotate.wav")

    def generate_clear_sound(self):
        """Generate sound for line clear."""
        duration = 0.3
        freqs = [440, 880]
        audio = sum(self.generate_sine_wave(f, duration) for f in freqs)
        audio = self.apply_envelope(audio, attack=0.02, decay=0.1)
        self.save_wav(audio, "clear.wav")

    def generate_hold_sound(self):
        """Generate sound for piece hold."""
        duration = 0.1
        freq = 400
        audio = self.generate_sine_wave(freq, duration)
        audio = self.apply_envelope(audio, attack=0.01, decay=0.05)
        self.save_wav(audio, "hold.wav")

    def generate_drop_sound(self):
        """Generate sound for hard drop."""
        duration = 0.15
        freq = 150
        audio = self.generate_sine_wave(freq, duration)
        audio = signal.square(2 * np.pi * freq * np.linspace(0, duration, int(self.sample_rate * duration)))
        audio = self.apply_envelope(audio, attack=0.01, decay=0.08)
        self.save_wav(audio, "drop.wav")

    def generate_game_over_sound(self):
        """Generate sound for game over."""
        duration = 1.0
        freqs = [220, 180, 140]
        audio = np.concatenate([
            self.generate_sine_wave(freq, duration/3) for freq in freqs
        ])
        audio = self.apply_envelope(audio, attack=0.05, decay=0.3)
        self.save_wav(audio, "game_over.wav")

    def generate_all_sounds(self):
        """Generate all Tetris sound effects."""
        self.generate_move_sound()
        self.generate_rotate_sound()
        self.generate_clear_sound()
        self.generate_hold_sound()
        self.generate_drop_sound()
        self.generate_game_over_sound()


def main():
    """Main function to generate all sound effects."""
    generator = TetrisSoundGenerator()
    generator.generate_all_sounds()


if __name__ == "__main__":
    main()