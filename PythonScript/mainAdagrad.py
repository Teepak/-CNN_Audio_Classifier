import feature_extractor
import modelAdagrad

# Path to the audio files directory
folder_path = 'recordings/'

# Extract audio features and save to CSV file
mfcc_features, spec_features = feature_extractor.feature_extractor(folder_path)
feature_extractor.write_to_csv(mfcc_features, spec_features, folder_path, 'dataset/mfcc_spec_features.csv')

#Train the model
modelAdagrad.train_model()

