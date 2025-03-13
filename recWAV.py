import sounddevice as sd
from scipy.io.wavfile import write

duration = 5  # seconds
sample_rate = 16000  # Whisper expects 16000 Hz

print("Recording...")
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
write("output.wav", sample_rate, recording)
print("Saved as output.wav")
