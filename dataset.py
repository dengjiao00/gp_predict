from torch.utils.data import dataset
from torch.utils.data.dataset import Dataset


class StockDataset(Dataset):
    def __init__(self, dataset_X, dataset_Y, transform=None, label_transform=None) -> None:
        super().__init__()
        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        assert self.dataset_X.shape[0] == self.dataset_Y.shape[0]
        return self.dataset_Y.shape[0]

    def __getitem__(self, index) -> tuple:
        X, Y = self.dataset_X[index], self.dataset_Y[index]
        if self.transform is not None:
            X = self.transform(X)

        if self.label_transform is not None:
            Y = self.label_transform(Y)

        return X, Y
