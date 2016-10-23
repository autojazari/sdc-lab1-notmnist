import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from zipfile import ZipFile

print('All modules imported.')

def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Path the file will be saved in
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

# Problem 1 - Implement a function that applies zero mean and equal variance scale to the data parameter
def normalize(data):
    return (data - data.mean()) / data.std()

def uncompress_features_labels(file):
    """
    Uncompress features and labels from zip file
    """
    
    features = []
    labels = []
    print("uncompressing", file)
    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

def save_data(data):
    # Save the data for easy access
    pickle_file = 'notMNIST.pickle'
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open('notMNIST.pickle', 'wb') as pfile:
                pickle.dump(data, pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    print('Data cached in pickle file.')


def one_hot_encode(train_labels, test_labels):
    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

def prep_notmnist():
    # does the picke already exist?
    if os.path.isfile('notMNIST.pickle'): return
    
    # Get the features and labels from the zip files
    
    # Download the training and test dataset.
    download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
    download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

    train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
    test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')
    print('All features and labels uncompressed.')

    train_features = normalize(train_features)
    test_features = normalize(test_features)

    one_hot_encode(train_labels, test_labels)
    print('Labels One-Hot Encoded')

    # Get randomized datasets for training and validation
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_features,
        train_labels,
        test_size=0.05,
        random_state=832289)
    print('Training features and labels randomized and split.')

    data = {
        'train_dataset': train_features,
        'train_labels': train_labels,
        'valid_dataset': valid_features,
        'valid_labels': valid_labels,
        'test_dataset': test_features,
        'test_labels': test_labels,
    }
    save_data(data)
