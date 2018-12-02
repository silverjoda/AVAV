import torchaudio
sound, sample_rate = torchaudio.load('audio_sources/music/audio_1.mp3', normalization=True)
print(sound.min(), sound.max())
torchaudio.save('foo_save.mp3', sound, sample_rate) # saves tensor to file