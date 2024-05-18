import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from onnxruntime.quantization import CalibrationDataReader

class MyDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def data_import_and_process(filename: str):
    # Load your dataset
    df = pd.read_csv(filename)

    # Convert diagnosis to binary
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    # Separate labels and features
    features = df.columns[2:]  # All columns except id and diagnosis

    # Normalize the data (using Min-Max normalization for simplicity)
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    return df

def prepare_dataset(df_path):
    df = data_import_and_process(df_path)

    # shuffle the data
    #shuffled_df = df.sample(frac=1).reset_index(drop=True)
    #df = shuffled_df

    # Save processed data
    df.to_csv(df_path.parent / f"preprocessed_data.csv", index=False)

    features = df.columns[2:]
    data = torch.tensor(df[features].values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)

    # Split the data: use the last 100 samples as test data, and the rest as training data
    data_train = data[150:]
    labels_train = labels[150:]
    
    data_calib = data[100:150]
    labels_calib = labels[100:150]
    
    data_test = data[:100]
    labels_test = labels[:100]

    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    
    return data_train, data_calib, data_test, labels_train, labels_calib, labels_test, train_loader

class CalibrationDataLoader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.idx = 0

    def get_next(self):
        if self.idx < len(self.data):
            current_data = self.data[self.idx]
            self.idx += 1
            return {'onnx::Gemm_0': [current_data]}
        else:
            return None