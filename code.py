!pip install -q openai-whisper transformers torch --index-url https://download.pytorch.org/whl/cu118
!pip install -q huggingface_hub

from getpass import getpass
from huggingface_hub import login

hf_token = getpass("👤 enter your HuggingFace API Token：")
login(hf_token)

print("✅ okay ")
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import files

#
uploaded = files.upload()
wav_path = next(iter(uploaded))

y, sr = librosa.load(wav_path, sr=16000)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
S_db = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel–Spectrogram')
plt.tight_layout()
plt.show()

#
class AccentCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Conv‐BN‐ReLU‐Pool
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128→64
        )
        # Conv‐BN‐ReLU‐Pool
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64→32
        )
        # Conv‐BN‐ReLU
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # [batch, 128, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # Dropout
        self.dropout = nn.Dropout(0.3)
        #
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)    # [B,128,1,1]
        x = x.view(x.size(0), -1)  # [B,128]
        x = self.dropout(x)
        return self.fc(x)

#
num_classes = 4
model = AccentCNN(num_classes)
mel_tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0)  # [1,1,128,T]
logits = model(mel_tensor.float())
probs = F.softmax(logits, dim=1).detach().numpy()[0]
labels = ['American','British','Australian','Indian']

print("output（no train，just for test）:")
for lbl, p in zip(labels, probs):
    print(f"  {lbl}: {p*100:5.1f}%")
