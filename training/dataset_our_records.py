# here we will import our own recorded files

import torchaudio
import IPython 
import matplotlib.pyplot as plt     




audio, sampling_rate = torchaudio.load("/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/test_python.wav")

audio = audio.mean(dim=0)   # ze względu na to, że audio jest stereo, a my chcemy mono, bierzemy średnią z obu kanałów

print("Audio:", audio)
print("Sampling Rate:", sampling_rate)
print("Total Samples in Audio:", len(audio))
print("Total Duration:", len(audio) / sampling_rate, "seconds")

IPython.display.Audio(audio, rate=16000)


plt.plot(audio)
plt.show()


