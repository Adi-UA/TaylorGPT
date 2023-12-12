import os
from pathlib import PosixPath

import dotenv
import torch
from lyricsgenius import Genius
from torch.utils.data import DataLoader, Dataset

# Load the environment variables
dotenv.load_dotenv()

# Set the data paths abd artist name
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN")
ARTIST_NAME = "Taylor Swift"
N_SONGS = 200
DATA_DIR = PosixPath(__file__).parent / "data"
RAW_DATA_PATH = DATA_DIR / "raw_taylor_swift_lyrics.txt"
DATA_PATH = DATA_DIR / "taylor_swift_lyrics.txt"
VOCAB_PATH = DATA_DIR / "vocab.txt"


####################
# Helper functions #
####################
def download_data():
    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize the Genius API client
    genius = Genius(GENIUS_API_TOKEN)
    genius.excluded_terms = ["(Remix)", "(Live)", "(Cover)"]
    genius.remove_section_headers = True
    genius.timeout = 15
    genius.sleep_time = 1e-3
    genius.retries = 5

    # Search for the artist
    artist = genius.search_artist(ARTIST_NAME, max_songs=N_SONGS, sort="popularity")

    # Save the data
    print(f"Saving data to {DATA_PATH}")
    with open(RAW_DATA_PATH, "w") as f:
        for song in artist.songs:
            lyrics = song.lyrics

            # remove first line
            try:
                lyrics = lyrics.split("\n", 1)[1]
            except IndexError:
                print(f"No contribution lines in {song.title}")
                pass

            f.write(lyrics)
            f.write("\n\n")


def build_vocab():
    with open(DATA_PATH, "r") as f:
        data = f.read()

    vocab = sorted(list(set(data)))

    print(f"Saving vocab to {VOCAB_PATH}")
    with open(VOCAB_PATH, "w") as f:
        f.write("\n".join(vocab))


def load_data():
    with open(DATA_PATH, "r") as f:
        data = f.read()

    return data


def train_test_split(data, train_frac):
    train_size = int(len(data) * train_frac)
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data


def load_vocab():
    with open(VOCAB_PATH, "r") as f:
        vocab = [line.rstrip("\n") for line in f]

    # Newline characters need some special handling
    vocab.append("\n")
    vocab = sorted(list(set(vocab)))
    vocab.remove("")

    return vocab


def get_encoder_decoder_fn(vocab):
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}

    encoder = lambda s: torch.tensor([char_to_idx[char] for char in s])
    decoder = lambda t: "".join([idx_to_char[idx] for idx in t])

    return encoder, decoder


######################
# Dataset and Dataloader #
######################
class CustomDataset(Dataset):
    def __init__(self, encoded_data, block_size):
        self.data = encoded_data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_dataloader(encoded_data, block_size, batch_size, shuffle):
    dataset = CustomDataset(encoded_data, block_size)
    num_workers = os.cpu_count()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader


if __name__ == "__main__":
    download_data()
    build_vocab()
