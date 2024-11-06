from openai import OpenAI
import librosa
import numpy as np

###tonality, pitch, tone analsyis

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

def summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration):
    summary = ""

    # Analyzing pitch variation
    if pitch_variation > 100:  # Thresholds are arbitrary; adjust based on your data
        summary += "The speaker shows a high pitch variation, indicating a dynamic and expressive tone throughout the speech.\n"
    elif pitch_variation > 50:
        summary += "The speaker maintains a moderate pitch variation, suggesting a balanced but engaging tone.\n"
    else:
        summary += "The speaker's pitch variation is low, indicating a more monotone and steady delivery style.\n"

    # Analyzing average volume
    if average_volume > 0.1:  # Thresholds are arbitrary; adjust based on your data
        summary += "The average volume is relatively high, reflecting a confident and assertive speaking style.\n"
    elif average_volume > 0.05:
        summary += "The average volume is moderate, suggesting a clear but not overly loud speaking style.\n"
    else:
        summary += "The average volume is low, which may indicate a softer or more subdued speaking style.\n"

    # Analyzing speaking rate
    if speaking_rate > 5:  # Thresholds are arbitrary; adjust based on your data
        summary += "The speaking rate is fast, possibly reflecting excitement or nervousness.\n"
    elif speaking_rate > 2:
        summary += "The speaking rate is moderate, indicating a balanced pace that is likely easy for listeners to follow.\n"
    else:
        summary += "The speaking rate is slow, which could indicate careful thought but may risk losing listener engagement.\n"

    # Analyzing pauses
    if pause_count > 10 and pause_duration > 1.0:  # Thresholds are arbitrary; adjust based on your data
        summary += "The speech contains frequent or lengthy pauses, suggesting a measured or contemplative pace.\n"
    elif pause_count > 5:
        summary += "The speech contains a moderate amount of pauses, adding a natural rhythm to the delivery.\n"
    else:
        summary += "The speech contains few pauses, which may make it sound continuous or uninterrupted.\n"

    return summary

print(summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration))


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
