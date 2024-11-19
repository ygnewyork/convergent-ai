""" 
Install this: ffmpeg -i input_video.mp4 -vn -acodec libmp3lame -q:a 4 output_audio.mp3
"""

import subprocess
import os
import librosa
import numpy as np

def extract_audio_from_video(video_path, output_audio_path):
    command = f"ffmpeg -i {video_path} -vn -acodec libmp3lame -q:a 4 {output_audio_path}"
    subprocess.call(command, shell=True)

def analyze_media(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.mp4', '.avi', '.mov']:  # Video files
        audio_path = file_path.rsplit('.', 1)[0] + '.mp3'
        extract_audio_from_video(file_path, audio_path)
        file_to_analyze = audio_path
    else:  # Audio files
        file_to_analyze = file_path

    # Your existing audio analysis code here
    y, sr = librosa.load(file_to_analyze)
    
    # ... (rest of your analysis code)

    return audio_score, summary

# Example usage
video_file = "interview.mp4"
score, summary = analyze_media(video_file)
print(f"Overall Score: {score}/100")
print(summary)
