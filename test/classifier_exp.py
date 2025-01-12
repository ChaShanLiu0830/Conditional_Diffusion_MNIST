# %%
import torch 
import matplotlib.pyplot as plt 
import pandas as pd 
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import copy
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
# %%
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)


def make_folder(path):
    if Path(path).exists() == False:
        Path(path).mkdir(parents=True)

class Trainer:
    def __init__(self, model, learning_rate=0.001, model_folder = "", model_name = "", device="cuda:0"):
        """
        Initialize the trainer with model, loss, and optimizer.
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.best_model_weights = None
        self.best_val_loss = float('inf')
        self.model_folder = model_folder
        self.model_name = model_name
        

    def train(self, train_loader, val_loader=None, epochs=5):
        """
        Train the model and validate after each epoch.
        """
        self.model = self.model.to(self.device)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for item in train_loader:
                images, labels = item['feature'].to(self.device), item['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Validation phase (if validation loader is provided)
            if val_loader:
                val_loss, val_correct, val_total = self.validate(val_loader)
                val_loss /= len(val_loader)
                val_accuracy = val_correct / val_total

                print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

                # Save the best model if validation loss improves
                if (val_loss < self.best_val_loss) & (epoch >= 10):
                    self.best_val_loss = val_loss
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                    self.save_best_model(f"{self.model_folder}/{self.model_name}.pt")
                    print(f"Best model updated with Val Loss: {val_loss:.4f}")

        # Load the best model weights after training
        if self.best_model_weights:
            self.model.load_state_dict(self.best_model_weights)
            print("Best model loaded.")

    def validate(self, val_loader):
        """
        Evaluate the model on the validation dataset.
        """
        self.model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for item in val_loader:
                images, labels = item['feature'].to(self.device), item['label'].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        return val_loss, val_correct, val_total

    def save_best_model(self, path="best_model.pth"):
        """
        Save the best model to the specified path.
        """
        if self.best_model_weights:
            torch.save(self.best_model_weights, path)
            print(f"Best model saved to {path}.")
        else:
            print("No best model to save.")

class Inferencer:
    def __init__(self, model, device = "cuda:0"):
        """Initialize the inferencer with the trained model."""
        self.model = model
        self.device = device

    def predict(self, dataloader):
        """Perform inference on a dataloader."""
        self.model.eval()
        predictions = []
        ground_truth = []
        with torch.no_grad():
            for item in dataloader:
                images, labels = item['feature'].to(self.device), item['label'].to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                predictions.append(preds.cpu().numpy())
                ground_truth.append(labels.cpu().numpy())
        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)
        return predictions, ground_truth

class GenMNIST(nn.Module):
    def __init__(self, data):
        self.data = torch.from_numpy(data)
        # len_data = self.data.shape[]
    def __getitem__(self, index):
        return {"feature": self.data[index, ...], "label": index % 10}
    def __len__(self):
        return len(self.data)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding = 1)  # Input channels=1, Output channels=32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding = 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust input size after convolution
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST digits

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) if x.dim() == 4 else  x.view(-1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Training a Deep Learning Model")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--diffusion_timestamp", type=int, default=0, help="diffusion timestamp")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--exp_name", type=str, default="guide_0.50", help="Path to the dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training (rest for validation)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--is_train", action="store_true" )
    parser.add_argument("--is_inference", action="store_true" )
    
    args = parser.parse_args()
    
    args.dataset_path = f"./gen_data/{args.exp_name}/{args.exp_name}"
    args.model_path = f"./weight/{args.exp_name}/"
    args.result_path = f"./result/{args.exp_name}/"
    
    args.model_name = f"model_t{args.diffusion_timestamp}"
    make_folder(args.dataset_path)
    make_folder(args.model_path)
    make_folder(args.result_path)
    
    print(args.dataset_path)
    data = np.concatenate([pd.read_pickle(f"{args.dataset_path}_{i}.pkl") for i in range(0, 10)], axis = 1)
    test_data = np.concatenate([pd.read_pickle(f"{args.dataset_path}_{i}.pkl") for i in range(10, 15)], axis = 1)
    print(test_data.shape)
    
    gen_dst = GenMNIST(data[args.diffusion_timestamp, ...].copy())
    gen_test_dst = GenMNIST(test_data[args.diffusion_timestamp, ...].copy())
    dst_size = len(gen_dst)
    indices = list(range(dst_size))
    
    train_indices, val_indices = train_test_split(indices, train_size=0.7, random_state=42)
    # val_indices, test_indices = train_test_split(val_indices, train_size=0.5, random_state=42)
    

    train_subset = Subset(gen_dst, train_indices)
    valid_subset = Subset(gen_dst, val_indices)
    train_dl = DataLoader(train_subset, shuffle=True, batch_size=1000)
    valid_dl = DataLoader(valid_subset, shuffle=True, batch_size=1000)
    test_dl = DataLoader(gen_test_dst, shuffle=True, batch_size=1000)
    
    model = Classifier().to(device)
    
    # trainer.train(train_dl, valid_dl, epochs = 120)
    if args.is_train:
        trainer = Trainer(model, 1e-4, model_folder = args.model_path, model_name= args.model_name)
        trainer.train(train_dl, valid_dl, epochs = 120)
    if args.is_inference:
        model.load_state_dict(torch.load(f"{args.model_path}/{args.model_name}.pt"))
        inferencer = Inferencer(model, device = device)
        prediction, label = inferencer.predict(test_dl)
        
        
        print(classification_report(label, prediction))
        result = []
        if Path(f"{args.result_path}/{args.exp_name}_result.pkl").exists():
            result = pd.read_pickle(f"{args.result_path}/{args.exp_name}_result.pkl")
        
        predict = classification_report(label, prediction, output_dict=True)
        dummy_result = {"diffusion_time": args.diffusion_timestamp}
        dummy_result.update({str(i):predict[str(i)]['precision'] for i in range(0, 10)})
        result.append(dummy_result)
        pd.to_pickle(result, f"{args.result_path}/{args.exp_name}_result.pkl")
        print(result)
        # result.append({})
        # # result = pd.DataFrame(classification_report(label, prediction, output_dict=True)).to_markdown()
        # with open(f"{args.result_path}/{args.exp_name}_t{args.diffusion_timestamp}.txt", "w") as f:
        # # You can customize the string format here if needed
        #     f.writelines(f"precision: {precision_score(label, prediction, average= 'micro')}\n")
        #     f.writelines(result)  # Exclude index for clean output
        
    
if __name__ == "__main__":
    main()
