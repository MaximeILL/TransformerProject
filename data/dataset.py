import torch
import torch.utils.data as data

def create_dataset(num_samples, seq_length, vocab_size):
    """
    Creates a dataset of random sequences for source and target data
    num_samples (int) : Number of samples to generate
    seq_length (int) : Length of each sequence
    vocab_size (int) : Size of the vocabulary (maximum value for random integers)
    returns Tuple[List[Tensor], List[Tensor]] : Source and target data sequences
    """
    src_data = []
    tgt_data = []
    for _ in range(num_samples):
        seq = torch.randint(2, vocab_size, (seq_length,))
        src_data.append(seq)
        tgt_data.append(seq.clone())
    return src_data, tgt_data

class TDataset(data.Dataset):
    """
    Custom PyTorch Dataset for handling source and target data
    src_data (List[Tensor]) : List of source sequences
    tgt_data (List[Tensor]) : List of target sequences
    """
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]
