import os
import librosa
import torch
import csv

def feature_extractor(folder_path):
    all_mfcc_features = []
    all_spec_features = []
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            # Load audio file
            y, sr = librosa.load(file_path, sr=22050)
            # Extract MFCC feature
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=512, n_mels=128)
            # Reshape MFCC to have a single feature vector for each audio file
            mfcc_tensor = torch.tensor(mfcc)
            mfcc_features = torch.mean(mfcc_tensor, dim=1).unsqueeze(0)

            # Extract spectrogram feature
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
            spectrogram_tensor = torch.tensor(spectrogram)
            spec_features = torch.mean(spectrogram_tensor, dim=1).unsqueeze(0)

            all_mfcc_features.append(mfcc_features)
            all_spec_features.append(spec_features)

    all_mfcc_features = torch.cat(all_mfcc_features, dim=0)
    all_spec_features = torch.cat(all_spec_features, dim=0)
    return all_mfcc_features, all_spec_features

def write_to_csv(mfcc_features, spec_features, folder_path, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['file'] + list(range(mfcc_features.shape[1])) + list(range(spec_features.shape[1])))
        for i, file in enumerate(os.listdir(folder_path)):
            if file.endswith('.wav'):
                writer.writerow([file] + mfcc_features[i].tolist() + spec_features[i].tolist())



