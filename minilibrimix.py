# !git clone 'https://github.com/microsoft/unilm'

# %cd unilm
# %cd wavlm

import torch
from WavLM import WavLM, WavLMConfig
import soundfile as sf
import glob

checkpoint = torch.load('/content/drive/MyDrive/SpeechTek/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()
model.double().to(device)

with torch.no_grad():
  for i, filename in enumerate(glob.glob('/content/drive/MyDrive/SpeechTek/miniLibrimix/**/*/*.wav')):
    print(i)
    sig, sr = sf.read(filename)
    t = torch.from_numpy(sig)
    t = t.unsqueeze(dim=0)
    t = t.double().to(device)
    rep = model.extract_features(t)[0]
    rep = rep.permute(0, 2, 1)
    print(rep.shape)
    torch.save(rep, filename.replace('.wav','.pt'))
