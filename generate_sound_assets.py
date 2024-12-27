#!/usr/bin/env python3
"""
/src/audio/generate_tetris_sounds.py

An enhanced script to generate polished Organic and Warm, Modern Arcade-style sound effects for a Tetris game using synthesized audio.
Generates WAV files for various game actions: move, rotate, clear, hold, drop, lock, and game over.

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
    """Generate and save polished Organic and Warm, Modern Arcade-style sound effects for Tetris game actions."""

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
            print(f"Created directory: {self.output_dir}")

    # --- Waveform Generators ---

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
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        wave = amplitude * np.sin(2 * np.pi * freq * t)
        return wave

    def generate_square_wave(self, freq: float, duration: float, amplitude: float = 1.0, duty: float = 0.5) -> np.ndarray:
        """
        Generate a square wave (or pulse wave with adjustable duty cycle).

        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            amplitude: Wave amplitude (default: 1.0)
            duty: Duty cycle for pulse wave (default: 0.5)

        Returns:
            numpy.ndarray: Audio samples
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        wave = amplitude * signal.square(2 * np.pi * freq * t, duty=duty)
        return wave

    def generate_triangle_wave(self, freq: float, duration: float, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a triangle wave.

        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            amplitude: Wave amplitude (default: 1.0)

        Returns:
            numpy.ndarray: Audio samples
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        # sawtooth(..., width=0.5) -> triangle wave
        wave = amplitude * signal.sawtooth(2 * np.pi * freq * t, width=0.5)
        return wave

    # --- Envelope Generator ---

    def apply_adsr_envelope(self, audio: np.ndarray, attack: float, decay: float, sustain_level: float, release: float) -> np.ndarray:
        """
        Apply an ADSR (Attack, Decay, Sustain, Release) envelope to the audio.

        Args:
            audio: Input audio samples
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain_level: Sustain level (0 to 1)
            release: Release time in seconds

        Returns:
            numpy.ndarray: Audio samples with ADSR envelope applied
        """
        total_samples = len(audio)
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_samples = total_samples - (attack_samples + decay_samples + release_samples)

        if sustain_samples < 0:
            # Adjust if total envelope exceeds audio length
            sustain_samples = 0
            decay_samples = max(int((decay / (attack + decay)) * total_samples), 1)
            attack_samples = total_samples - decay_samples

        # Construct ADSR
        envelope = np.concatenate([
            np.linspace(0, 1, attack_samples),                        # Attack
            np.linspace(1, sustain_level, decay_samples),             # Decay
            np.full(sustain_samples, sustain_level),                  # Sustain
            np.linspace(sustain_level, 0, release_samples)            # Release
        ])

        # Match audio length
        if len(envelope) < len(audio):
            envelope = np.pad(envelope, (0, len(audio) - len(envelope)), 'constant')
        else:
            envelope = envelope[:len(audio)]

        return audio * envelope

    # --- Effects ---

    def multi_stage_reverb(self, audio: np.ndarray, stages=3, base_decay=0.15, mix=0.3) -> np.ndarray:
        """
        Apply a multi-stage reverb by convolving multiple short impulses to simulate a smoother decay.

        Args:
            audio: Input audio samples
            stages: Number of reverb stages
            base_decay: Base time for each impulse
            mix: Wet/Dry mix ratio (0 to 1)

        Returns:
            numpy.ndarray: Audio samples with multi-stage reverb applied
        """
        # Start with dry signal
        wet_signal = np.zeros_like(audio)
        for stage in range(1, stages + 1):
            ir_length = int(self.sample_rate * (base_decay * stage))
            # Exponential impulse
            impulse = np.exp(-np.linspace(0, stage, ir_length))
            impulse /= np.max(impulse)
            # Convolve
            convolved = np.convolve(audio, impulse, mode='full')[:len(audio)]
            wet_signal += convolved * (1.0 / stages)

        return (1 - mix) * audio + mix * wet_signal

    def harmonic_exciter(self, audio: np.ndarray, amount: float = 0.1, freq_split: float = 1000.0) -> np.ndarray:
        """
        Add subtle harmonic excitement to higher frequencies.

        Args:
            audio: Input audio samples
            amount: Exciter amount (0 to 1)
            freq_split: Crossover frequency in Hz (above which excitement is applied)

        Returns:
            numpy.ndarray: Audio samples with enhanced higher frequencies
        """
        # Split frequencies using a simple high-pass filter at freq_split
        nyquist = 0.5 * self.sample_rate
        highpass_cutoff = freq_split / nyquist
        b, a = signal.butter(2, highpass_cutoff, btype='high')
        high_freq = signal.lfilter(b, a, audio)

        # Apply soft distortion to the high frequencies
        excited = np.tanh(high_freq * (1.0 + amount * 5.0))

        # Merge back with the low frequencies
        low_freq = audio - high_freq
        return low_freq + excited * amount

    def simple_eq(self, audio: np.ndarray, low_gain: float = 1.0, mid_gain: float = 1.0, high_gain: float = 1.0) -> np.ndarray:
        """
        Apply a simplistic multi-band filter to shape low, mid, and high frequencies.

        Args:
            audio: Input audio samples
            low_gain: Gain for the low band
            mid_gain: Gain for the mid band
            high_gain: Gain for the high band

        Returns:
            numpy.ndarray: Audio samples with basic multi-band EQ applied
        """
        nyquist = 0.5 * self.sample_rate

        # Define cutoff frequencies (example: 300 Hz and 3000 Hz)
        low_cutoff = 300.0 / nyquist
        high_cutoff = 3000.0 / nyquist

        # Low band (lowpass)
        b_low, a_low = signal.butter(2, low_cutoff, btype='low')
        low_band = signal.lfilter(b_low, a_low, audio) * low_gain

        # High band (highpass)
        b_high, a_high = signal.butter(2, high_cutoff, btype='high')
        high_band = signal.lfilter(b_high, a_high, audio) * high_gain

        # Mid band (bandpass)
        b_mid, a_mid = signal.butter(2, [low_cutoff, high_cutoff], btype='band')
        mid_band = signal.lfilter(b_mid, a_mid, audio) * mid_gain

        # Combine
        return low_band + mid_band + high_band

    def stereo_panning(self, audio: np.ndarray, pan: float = 0.0) -> np.ndarray:
        """
        Apply stereo panning to the audio.

        Args:
            audio: Mono audio samples
            pan: Panning value between -1.0 (left) and 1.0 (right). 0.0 is center.

        Returns:
            numpy.ndarray: Stereo audio samples
        """
        left = np.sqrt(0.5 * (1.0 - pan))
        right = np.sqrt(0.5 * (1.0 + pan))
        stereo_audio = np.vstack((audio * left, audio * right)).T
        return stereo_audio

    # --- Utility Methods ---

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to prevent clipping.

        Args:
            audio: Audio samples

        Returns:
            numpy.ndarray: Normalized audio samples
        """
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio
        return audio / max_val

    def save_wav(self, audio: np.ndarray, filename: str):
        """
        Save audio data as a WAV file.

        Args:
            audio: Audio samples to save
            filename: Output filename
        """
        audio = self.normalize_audio(audio)

        # Convert to 16-bit PCM
        if audio.ndim == 1:
            audio_16bit = (audio * 32767).astype(np.int16)
        elif audio.ndim == 2:
            audio_16bit = (audio * 32767).astype(np.int16)
        else:
            raise ValueError("Audio array has unsupported number of dimensions.")

        filepath = os.path.join(self.output_dir, filename)
        wavfile.write(filepath, self.sample_rate, audio_16bit)
        print(f"Generated: {filepath}")

    # --- Sound Generators ---

    def generate_move_sound(self):
        """
        Enhanced Move Sound:
        - Short, punchy, mid-frequency percussive tone with subtle layering and quick envelope.
        """
        duration = 0.1
        print("Generating move.wav...")

        # Layer 1: Organic - Low frequency sine
        freq_organic = 300
        organic = self.generate_sine_wave(freq_organic, duration, amplitude=0.5)

        # Layer 2: Modern Arcade - Higher frequency square (slightly detuned)
        freq_modern = 305
        modern = self.generate_square_wave(freq_modern, duration, amplitude=0.4, duty=0.4)

        combined = organic + modern

        # Apply ADSR
        adsr = self.apply_adsr_envelope(combined, attack=0.005, decay=0.02, sustain_level=0.3, release=0.02)

        # EQ to tighten low end and emphasize mid-range
        eqed = self.simple_eq(adsr, low_gain=0.8, mid_gain=1.1, high_gain=1.0)

        # Multi-stage reverb for a subtle sense of space
        reverb_applied = self.multi_stage_reverb(eqed, stages=2, base_decay=0.1, mix=0.1)

        # Harmonic exciter for a slight brightening
        excited = self.harmonic_exciter(reverb_applied, amount=0.05, freq_split=800.0)

        # Pan slightly left
        stereo = self.stereo_panning(excited, pan=-0.1)
        self.save_wav(stereo, "move.wav")

    def generate_rotate_sound(self):
        """
        Enhanced Rotate Sound:
        - Bell-like chime plus airy pad, gentle sweep, subtle reverb and exciter.
        """
        duration = 0.15
        print("Generating rotate.wav...")

        # Layer 1: Organic - Triangle wave
        freq_organic = 700
        organic = self.generate_triangle_wave(freq_organic, duration, amplitude=0.4)

        # Layer 2: Modern Arcade - Slightly detuned sine pad
        freq_modern = 710
        pad = self.generate_sine_wave(freq_modern, duration, amplitude=0.3)

        combined = organic + pad

        # ADSR
        adsr = self.apply_adsr_envelope(combined, attack=0.02, decay=0.04, sustain_level=0.5, release=0.06)

        # EQ
        eqed = self.simple_eq(adsr, low_gain=0.9, mid_gain=1.1, high_gain=1.1)

        # Reverb
        reverb_applied = self.multi_stage_reverb(eqed, stages=3, base_decay=0.12, mix=0.2)

        # Exciter
        excited = self.harmonic_exciter(reverb_applied, amount=0.1, freq_split=1200.0)

        # Stereo - centered
        stereo = self.stereo_panning(excited, pan=0.0)
        self.save_wav(stereo, "rotate.wav")

    def generate_clear_sound(self):
        """
        Enhanced Clear Sound:
        - Uplifting chord-like hit, bright, celebratory, layered with subtle sparkle.
        """
        duration = 0.3
        print("Generating clear.wav...")

        # Layer 1: Organic - Sine wave at fundamental
        freq_organic = 1000
        organic = self.generate_sine_wave(freq_organic, duration, amplitude=0.4)

        # Layer 2: Modern Arcade - Square wave at a harmonic
        freq_modern = 1500
        modern = self.generate_square_wave(freq_modern, duration, amplitude=0.3, duty=0.5)

        # Additional sparkle layer (very quiet)
        freq_sparkle = 2000
        sparkle = self.generate_sine_wave(freq_sparkle, duration, amplitude=0.15)

        combined = organic + modern + sparkle

        # ADSR
        adsr = self.apply_adsr_envelope(combined, attack=0.005, decay=0.05, sustain_level=0.7, release=0.05)

        # EQ
        eqed = self.simple_eq(adsr, low_gain=0.9, mid_gain=1.0, high_gain=1.2)

        # Reverb
        reverb_applied = self.multi_stage_reverb(eqed, stages=3, base_decay=0.15, mix=0.4)

        # Exciter
        excited = self.harmonic_exciter(reverb_applied, amount=0.1, freq_split=1500.0)

        # Stereo - slightly right
        stereo = self.stereo_panning(excited, pan=0.1)
        self.save_wav(stereo, "clear.wav")

    def generate_hold_sound(self):
        """
        Enhanced Hold Sound:
        - Smooth, sustained pad with gentle attack, warm resonance, and subtle reverb.
        """
        duration = 0.2
        print("Generating hold.wav...")

        # Layer 1: Organic - Sine wave
        freq_organic = 500
        organic = self.generate_sine_wave(freq_organic, duration, amplitude=0.5)

        # Layer 2: Modern Arcade - Soft square wave
        freq_modern = 510
        modern = self.generate_square_wave(freq_modern, duration, amplitude=0.3, duty=0.4)

        combined = organic + modern

        # ADSR
        adsr = self.apply_adsr_envelope(combined, attack=0.05, decay=0.1, sustain_level=0.8, release=0.2)

        # EQ
        eqed = self.simple_eq(adsr, low_gain=1.0, mid_gain=1.0, high_gain=0.9)

        # Reverb
        reverb_applied = self.multi_stage_reverb(eqed, stages=3, base_decay=0.2, mix=0.2)

        # Harmonic exciter
        excited = self.harmonic_exciter(reverb_applied, amount=0.05, freq_split=900.0)

        # Stereo
        stereo = self.stereo_panning(excited, pan=0.0)
        self.save_wav(stereo, "hold.wav")

    def generate_drop_sound(self):
        """
        Enhanced Drop Sound:
        - Punchy low-end impact, short envelope, minimal reverb to maintain clarity.
        """
        duration = 0.2
        print("Generating drop.wav...")

        # Layer 1: Organic - Low square wave
        freq_organic = 200
        organic = self.generate_square_wave(freq_organic, duration, amplitude=0.5, duty=0.4)

        # Layer 2: Modern Arcade - Slightly higher frequency square
        freq_modern = 220
        modern = self.generate_square_wave(freq_modern, duration, amplitude=0.3, duty=0.4)

        combined = organic + modern

        # ADSR
        adsr = self.apply_adsr_envelope(combined, attack=0.002, decay=0.04, sustain_level=0.2, release=0.02)

        # EQ
        eqed = self.simple_eq(adsr, low_gain=1.2, mid_gain=1.0, high_gain=0.8)

        # Reverb (very low mix)
        reverb_applied = self.multi_stage_reverb(eqed, stages=2, base_decay=0.08, mix=0.1)

        # Harmonic exciter for a slight punch in upper-lows
        excited = self.harmonic_exciter(reverb_applied, amount=0.08, freq_split=400.0)

        # Stereo
        stereo = self.stereo_panning(excited, pan=0.0)
        self.save_wav(stereo, "drop.wav")

    def generate_lock_sound(self):
        """
        Enhanced Lock Sound:
        - Firm metallic click with quick envelope, emphasizing mid-range for clarity.
        """
        duration = 0.15
        print("Generating lock.wav...")

        # Layer 1: Organic - Triangle wave
        freq_organic = 500
        organic = self.generate_triangle_wave(freq_organic, duration, amplitude=0.5)

        # Layer 2: Modern Arcade - Slightly detuned square wave
        freq_modern = 505
        modern = self.generate_square_wave(freq_modern, duration, amplitude=0.3, duty=0.5)

        combined = organic + modern

        # ADSR
        adsr = self.apply_adsr_envelope(combined, attack=0.005, decay=0.04, sustain_level=0.2, release=0.02)

        # EQ
        eqed = self.simple_eq(adsr, low_gain=0.9, mid_gain=1.2, high_gain=1.0)

        # Reverb (subtle)
        reverb_applied = self.multi_stage_reverb(eqed, stages=2, base_decay=0.1, mix=0.1)

        # Exciter
        excited = self.harmonic_exciter(reverb_applied, amount=0.05, freq_split=1000.0)

        # Stereo - slightly left
        stereo = self.stereo_panning(excited, pan=-0.1)
        self.save_wav(stereo, "lock.wav")

    def generate_game_over_sound(self):
        """
        Enhanced Game Over Sound:
        - Ominous, dramatic buildup with evolving pad, deep low end, extended reverb tail.
        """
        duration = 0.8
        print("Generating game_over.wav...")

        # Layer 1: Organic - Low sine wave
        freq_organic = 150
        organic = self.generate_sine_wave(freq_organic, duration, amplitude=0.4)

        # Layer 2: Modern Arcade - Descending sweep
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        freq_sweep = 200 - 100 * (t / duration)  # Sweep from 200 Hz down to 100 Hz
        sweep = 0.3 * np.sin(2 * np.pi * freq_sweep * t)

        combined = organic + sweep

        # ADSR
        adsr = self.apply_adsr_envelope(combined, attack=0.1, decay=0.4, sustain_level=0.3, release=0.4)

        # EQ
        eqed = self.simple_eq(adsr, low_gain=1.1, mid_gain=0.95, high_gain=0.9)

        # Multi-stage reverb for a long tail
        reverb_applied = self.multi_stage_reverb(eqed, stages=4, base_decay=0.2, mix=0.5)

        # Harmonic exciter for a haunting shimmer
        excited = self.harmonic_exciter(reverb_applied, amount=0.08, freq_split=600.0)

        # Stereo
        stereo = self.stereo_panning(excited, pan=0.0)
        self.save_wav(stereo, "game_over.wav")

    def generate_all_sounds(self):
        """Generate all polished Tetris sound effects."""
        print("Starting generation of all enhanced Tetris sound effects...")
        self.generate_move_sound()
        self.generate_rotate_sound()
        self.generate_clear_sound()
        self.generate_hold_sound()
        self.generate_drop_sound()
        self.generate_lock_sound()
        self.generate_game_over_sound()
        print("All sound effects generated successfully.")


def main():
    """Main function to generate all enhanced Tetris sound effects."""
    generator = TetrisSoundGenerator()
    generator.generate_all_sounds()


if __name__ == "__main__":
    main()
