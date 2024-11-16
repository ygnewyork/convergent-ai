import librosa
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# Load the audio file
audio_path = "yash_sample.mp3"
y, sr = librosa.load(audio_path)

### 1. Improved Pitch Extraction
# Extract pitch using YIN (more robust for speech analysis)
fmin = 85  # Minimum frequency for human speech
fmax = 255  # Maximum frequency for human speech
pitches = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)

# Filter out zero or non-speech values (YIN avoids most, but this ensures cleanup)
pitch_values = [p for p in pitches if fmin <= p <= fmax]

# Calculate pitch variation
pitch_variation = np.var(pitch_values)
print("Pitch Variation:", pitch_variation)

### 2. Improved Loudness (RMS) and Pause Detection
# Calculate RMS for loudness
rms = librosa.feature.rms(y=y)[0]

# Smooth the RMS values to reduce fluctuations
smoothed_rms = scipy.ndimage.uniform_filter1d(rms, size=10)

# Define a more robust silence threshold
silence_threshold = 0.1 * smoothed_rms.max()

# Detect pauses as RMS values below the threshold
pauses = smoothed_rms < silence_threshold
pause_count = np.sum(pauses)
pause_duration = np.sum(pauses) / sr

print("Number of Pauses:", pause_count)
print("Total Pause Duration (seconds):", pause_duration)

### 3. Speaking Rate Calculation
# Detect onsets to approximate syllables
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# Calculate speaking rate as syllables per second
speaking_rate = len(onset_times) / librosa.get_duration(y=y, sr=sr)
print("Speaking Rate (syllables/sec):", speaking_rate)

### 4. Visualization for Debugging
# Plot RMS values and silence threshold
plt.figure(figsize=(10, 4))
plt.plot(smoothed_rms, label='Smoothed RMS')
plt.axhline(silence_threshold, color='r', linestyle='--', label='Silence Threshold')
plt.title('Smoothed RMS Over Time')
plt.legend()
plt.show()

# Plot pitch values
plt.figure(figsize=(10, 4))
plt.plot(pitch_values, label='Pitch Values')
plt.title('Pitch Over Time')
plt.legend()
plt.show()

### Updated Summary Function
def summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration):
    summary = ""

    # Analyze pitch variation (ideal range: 30–100)
    if pitch_variation > 100:
        summary += "The speaker shows a very high pitch variation, indicating a dynamic and expressive tone. It might sound overly dramatic.\n"
    elif 30 <= pitch_variation <= 100:
        summary += "The speaker maintains an ideal pitch variation, suggesting a balanced and engaging tone.\n"
    else:
        summary += "The speaker's pitch variation is low, indicating a monotone delivery style that might affect engagement.\n"

    # Analyze average volume (ideal: 0.05–0.15)
    if average_volume > 0.15:
        summary += "The average volume is high, reflecting a confident and assertive speaking style. It could risk sounding overly loud.\n"
    elif 0.05 <= average_volume <= 0.15:
        summary += "The average volume is ideal, suggesting clarity and an approachable delivery.\n"
    else:
        summary += "The average volume is low, which may indicate a subdued style or lack of projection.\n"

    # Analyze speaking rate (ideal: 2–4 syllables/sec)
    if speaking_rate > 4:
        summary += "The speaking rate is fast, possibly reflecting excitement or nervousness. Slowing down might help clarity.\n"
    elif 2 <= speaking_rate <= 4:
        summary += "The speaking rate is ideal, indicating a balanced pace that is easy to follow.\n"
    else:
        summary += "The speaking rate is slow, which could indicate careful thought but might risk losing engagement.\n"

    # Analyze pauses (ideal: 5–10 pauses per minute, total pause duration ≤ 20% of speech time)
    total_duration = librosa.get_duration(y=y, sr=sr)
    pauses_per_minute = (pause_count / total_duration) * 60
    pause_percentage = (pause_duration / total_duration) * 100

    if pauses_per_minute > 10 or pause_percentage > 20:
        summary += "The speech contains frequent or lengthy pauses, which might disrupt the natural flow.\n"
    elif 5 <= pauses_per_minute <= 10 and pause_percentage <= 20:
        summary += "The speech contains an ideal number and duration of pauses, adding a natural rhythm and allowing for emphasis.\n"
    else:
        summary += "The speech contains few pauses, making it sound continuous or uninterrupted. Adding pauses might improve engagement.\n"

    return summary

# Print updated summary
average_volume = smoothed_rms.mean()
print(summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration))
