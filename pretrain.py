import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
# Load the data

device = torch.device('cuda')


data = pd.read_csv(r'/home/dl/PycharmProjects/8_30/filtered_data_9.17.csv')

data['RESULT_DATE_TIME'] = pd.to_datetime(data['RESULT_DATE_TIME'])
data['time_interval'] = data.groupby(['PATIENT_ID', 'VISIT_ID'])['RESULT_DATE_TIME'].diff().dt.total_seconds().fillna(0)

# Calculate the mean time interval for each patient
patient_mean_intervals = data.groupby('PATIENT_ID')['time_interval'].mean()

data = data.merge(patient_mean_intervals, on='PATIENT_ID', suffixes=('', '_mean_per_patient'))
## 此处计算每一组的时间间隔平均值，并且重复放到数据组最后一列，
# 即新属性time_interval_mean_per_patient


# Data preprocessing
X = data.drop('白蛋白', axis=1)
y = data['白蛋白']


# Split the data
grouped_data = list(X.groupby(['PATIENT_ID', 'VISIT_ID']))
# train_groups, val_groups = train_test_split(grouped_data, test_size=0.2, random_state=42)

def extract_data_from_groups(groups):
    X_list = []
    y_list = []
    for _, group in groups:
        X_list.append(group)
        y_list.append(y[group.index])
    return pd.concat(X_list), pd.concat(y_list)

X_train, y_train = extract_data_from_groups(grouped_data)


# X_val, y_val = extract_data_from_groups(val_groups)

# Drop non-feature columns
columns_to_drop = ['PATIENT_ID', 'VISIT_ID', 'RESULT_DATE_TIME', 'RELEVANT_CLINIC_DIAG']
X_train = X_train.drop(columns=columns_to_drop)
# X_val = X_val.drop(columns=columns_to_drop)

# Identify non-numeric columns
non_numeric_columns = X_train.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

# Drop non-numeric columns
X_train = X_train.drop(columns=non_numeric_columns)
# X_val = X_val.drop(columns=non_numeric_columns)

# Standardize the data
scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

X_train_scaled = scaler.fit_transform(X_train)  # Exclude original_index from scaling
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()  # Convert back to 1D array after scaling

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim,d_model, nhead, num_layers ,max_seq_length):
        super(TransformerRegressor, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)  # Simple linear embedding
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead,   dim_feedforward=32,dropout=0.1),num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)

    def forward(self, src, use_positional_encoding=False):
        if use_positional_encoding:
            # Create positional encodings
            positions = (torch.arange(0, src.size(1)).expand(src.size(0), -1)).to(src.device)
            src = self.embedding(src) + self.pos_encoder(positions)
        else:

            src = self.embedding(src)

        # Pass through transformer
        output = self.transformer(src)

        # Pass through final FC layer
        return self.fc(output)

# 2. Pretrain the Transformer model
# Load the provided data
# X_pretrain_transformer_tensor, y_pretrain_transformer_tensor = ...

# Reshape the data for the Transformer and convert to PyTorch tensors
X_pretrain_transformer_tensor= torch.tensor((X_train_scaled), dtype=torch.float32).to(device)
# X_pretrain_transformer_tensor = X_pretrain_transformer_tenso.unsqueeze(1)  # Add the sequence_length dimension

y_pretrain_transformer_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
# X_pretrain_transformer_tensor = torch.FloatTensor(X_pretrain_train_array).unsqueeze(1)
# y_pretrain_transformer_tensor = torch.FloatTensor(y_pretrain_train_array).unsqueeze(1)

# Model parameters
d_model = 16
nhead = 4
num_layers = 4
input_dim = X_pretrain_transformer_tensor.shape[1]

max_seq_length = 1

# Create the model, loss function, and optimizer
model = TransformerRegressor(input_dim,d_model, nhead, num_layers,  max_seq_length)
model = model.to('cuda')






criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training configurations
num_epochs = 50






# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(X_pretrain_transformer_tensor)
    loss = criterion(predictions, y_pretrain_transformer_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")

# 3. Save the pretrained model weights
torch.save(model.state_dict(), "pretrained_transformer_alb_9.17.pth")

# 4. Finetune on the larger irregular dataset
# Load the irregular dataset
# X_irregular, y_irregular = ...

# Dataset and DataLoader for batching
# dataset = TensorDataset(X_irregular, y_irregular)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# # Load the pretrained weights into the model
# model.load_state_dict(torch.load("pretrained_transformer.pth"))
#
# # Finetuning loop
# for epoch in range(num_epochs):
#     for batch_X, batch_y in loader:
#         optimizer.zero_grad()
#         predictions = model(batch_X)
#         loss = criterion(predictions, batch_y)
#         loss.backward()
#         optimizer.step()
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")

# Example of using the modified model:
# For pretraining:
# predictions = model(data, use_positional_encoding=False)

# For finetuning on irregular data:
# predictions = model(data, use_positional_encoding=True)


# Define the ModifiedTrans model
# Calculate the mean time interval from the training data