import random
import torch
import librosa
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.transform import resize

# Creating Dataset Class
class custom_dataset(Dataset):
    """
    Custom dataset class for audio classification tasks.

    Args:
        dataframe: Pandas DataFrame containing file paths and labels.
        sr: Sampling rate of audio files.
        segment_duration: Duration of audio segments in seconds.
        img_height: Height of the resized mel spectrogram image.
        img_width: Width of the resized mel spectrogram image.
        transform: EfficientNet transforms for image preprocessing.

    Returns:
        Tuple containing the resized mel spectrogram image and the label.
    """

    def __init__(self, dataframe, sr=22050, segment_duration=5, img_height=288, img_width=288, transform=None):
        self.dataframe = dataframe
        self.labels = list(dataframe["Class"])
        self.filepaths = list(dataframe["FilePath"])
        self.sr = sr
        self.segment_duration = segment_duration
        self.img_height = img_height
        self.img_width = img_width
        self.total_duration = 30
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of data samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Loads and processes an audio file based on the given index.

        Args:
            index: Index of the data sample.

        Returns:
            Tuple containing the resized mel spectrogram image and the label.
        """
        file_path = self.filepaths[index]
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)

        # --- Random 5s segment ---
        max_offset = self.total_duration - self.segment_duration
        offset = random.uniform(0, max_offset)

        signal, sr = librosa.load(file_path, sr=self.sr, offset=offset, duration=self.segment_duration)

        # --- Mel Spectrogram ---
        spec = librosa.feature.melspectrogram(y=signal,
                                            sr=sr,
                                            n_fft=2048,
                                            hop_length=512,
                                            n_mels=self.img_height)
        spec_db = librosa.power_to_db(spec, ref=np.max)

        # --- Resize ---
        spec_resized = resize(spec_db, (self.img_height, self.img_width), anti_aliasing=True)

        # --- Convert to RGB (3-channel) ---
        spec_resized = np.stack([spec_resized] * 3, axis=-1)
        spec_resized = (255 * (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min())).astype(np.uint8)

        # --- Convert to PIL Image ---
        spec_img = Image.fromarray(spec_resized)

        # --- Apply EfficientNet transforms ---
        if self.transform:
            audio = self.transform(spec_img)
        else:
            audio = transforms.ToTensor()(spec_img)

        return audio, label
    

def get_csv(data_path: Path):

    """
    Extracts audio file paths and their corresponding genre labels from the data directory
    and saves them to a CSV file.

    Args:
        data_path (Path): The path to the directory containing the audio files.

    Returns:
        tuple: A tuple containing the pandas DataFrame and a list of class names.
    """

    # Getting filepaths and labels from the dataset
    data_dir = data_path

    filepaths = []
    labels = []

    for genre_dir in data_dir.iterdir():
        if genre_dir.is_dir():
            genre = genre_dir.name

            for file in genre_dir.glob("*.wav"):
                filepaths.append(str(file))
                labels.append(genre)

    # Creating Dataframe
    df = pd.DataFrame({"FilePath": filepaths,
                    "Class": labels})

    # Mapping Gneres to numbers
    df["Class"] = df["Class"].astype("category").cat.codes

    # Saving to CSV
    df.to_csv("dataset.csv", index=False)
    class_names = list(set(labels))
    print(f"[INFO] Saved the `dataset.csv` in current directory")
    print(df.head())
    
    return df, class_names



def create_dataloaders(dataset: Dataset,
                       batch_size: int,
                       test_size: float,
                       num_workers: int=0):
    
    """
    Creates PyTorch DataLoader instances for training and testing.

    Args:
        dataset (Dataset): The dataset to use for training and testing.
        batch_size (int): The batch size to use for the data loaders.
        test_size (float): The proportion of the dataset to use for testing.
        num_workers (int, optional): The number of worker processes to use for data loading.
                                     Defaults to 0.

    Returns:
        tuple: A tuple containing the training DataLoader and the testing DataLoader.
    """
    
    # Creating train test splits
    test_size = int(test_size * len(dataset))   #type: ignore
    train_size = len(dataset) - test_size       #type: ignore
    train_data, test_data = random_split(dataset=dataset,
                                         lengths=[train_size, test_size])
    
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    
    return train_dataloader, test_dataloader

