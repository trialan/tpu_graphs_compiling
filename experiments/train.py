import torch
from torch_geometric.data import DataLoader

from gnn import KernelGNN
from data_gnn import get_tile_data

device = 'cpu'
num_epochs = 3
batch_size = 32
learning_rate = 0.001

training_data = get_tile_data('train')
validation_data = get_tile_data('valid')

exclude_keys = ["edge_index",
                "x"] #don't all have same number of edges and vertices
train_loader = DataLoader(training_data,
                          batch_size=batch_size,
                          exclude_keys=exclude_keys,
                          shuffle=True)
valid_loader = DataLoader(validation_data,
                          batch_size=batch_size,
                          exclude_keys=exclude_keys,
                          shuffle=False)


node_feature_dim = 140
config_feature_dim = 24
model = KernelGNN(node_feature_dim, config_feature_dim)
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)
# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:  # Iterate over each batch in the DataLoader
        data = data.to(device)
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Perform a forward pass
        loss = criterion(output, data.y)  # Compute the loss
        loss.backward()  # Perform a backward pass
        optimizer.step()  # Update the weights
        running_loss += loss.item() * data.num_graphs

    avg_train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    # Validation step
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        for data in valid_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            running_val_loss += loss.item() * data.num_graphs

        avg_val_loss = running_val_loss / len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    if epoch == 0 or avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"NEW BEST val. loss: {round(best_val_loss, 6)}")
        #torch.save(model.state_dict(), 'best_model.pth')
        #print("Saved best model")

print("Training complete")
