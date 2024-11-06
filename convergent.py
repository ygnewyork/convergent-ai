from openai import OpenAI
import librosa
import numpy as np


audio_path = ("yash_sample.mp3")
y, sr = librosa.load(audio_path)

# Extract pitch using librosa's YIN algorithm
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

# Extracting pitch values from piptrack
pitch_values = []
for t in range(pitches.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch = pitches[index, t]
    if pitch > 0:  # Filter out zero values
        pitch_values.append(pitch)
pitch_variation = np.var(pitch_values)
print("Pitch Variation:", pitch_variation)


# Calculate the Root Mean Square (RMS) for loudness
rms = librosa.feature.rms(y=y)
average_volume = rms.mean()

print("Average Volume (RMS):", average_volume)

# Detect onsets, which can be used to estimate syllable rate
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# Calculate speaking rate as syllables per second
speaking_rate = len(onset_times) / librosa.get_duration(y=y, sr=sr)
print("Speaking Rate (syllables/sec):", speaking_rate)

# Define a threshold for silence (e.g., 20% of max RMS)
silence_threshold = 0.2 * rms.max()

# Detect pauses as RMS values below the threshold
pauses = rms[0] < silence_threshold
pause_count = np.sum(pauses)
pause_duration = np.sum(pauses) / sr  # Total duration of pauses

print("Number of Pauses:", pause_count)
print("Total Pause Duration (seconds):", pause_duration)

##audio-to-text translation
client = OpenAI()
audio_file= open("yash_sample.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)

##gpt integration
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a text analyzer who specializes in scrutinizing job descriptions and seeing if the answers somebody gives to potential interview questions align with the job's values. You will be given a job description, interview question, and someone giving a sample response. Your job is to evaluate how well the response answers the question and aligns with the job description. Always give constructive criticsm to the user."},
        {
            "role": "user",
            "content": "Job Description: We're not rockstars or ninjas, just regular coders trying to make it a little easier to manage a doctor's office." +
            "Working in Ruby/Rails API with AngularJS/React front-end deployed on AWS cloud infrastructure we help keep track of patient appointments, drug inventory, and integrate with systems to send appointment reminders and bill insurance claims. " + 
            "We hire people we can trust. As soon as you hit merge, your code is headed to production (and ringing a gong in our Austin office) and you're given a wide berth to solve problems and improve our engineering team. " +
            "Come join a 20-person engineering organization spread across three smaller feature teams where you can make a direct impact." +
            "Our Stack React/ES6 Front-end - Ruby on Rails API - MySQL - AWS - Docker What You'll Do Write and deploy code - Create tests - Work with product on stories - Improve our processes - Attend daily stand-ups - Stay curious and enjoy learning new things " +
            "Desired Skills/Experience 4+ years Ruby, Python, Java, Javascript or other language - Strong problem-solving skills - Must be authorized to work in the United States About WeInfuse Come join our growing company. " +
            "We are an established healthcare SaaS start-up with offices in Dallas and Austin. Founded in 2016, WeInfuse is an infusion center software and consulting organization." +
            "Our founders and their team have developed the first and only end-to-end software solution for infusion centers that has gained significant traction in the market. " +
            "In addition to providing the industry’s leading SaaS solution, WeInfuse provides infusion center start-up, optimization and pharmaceutical manufacturer consulting services. " +
            "Interview Question: What skills do you have that will be relevant to this position?" +
            "User Response:" + transcription.text
        }
    ]
)
print(completion.choices[0].message)
