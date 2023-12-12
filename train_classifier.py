import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
from datetime import datetime
import shutil
import numpy as np

model_save_dir = 'Trained_Models'
os.makedirs(model_save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the paths and split ratios
source_dir = 'food-101/food-101/food-101/images'  # Replace with source directory path
destination_dir = 'data'  # Replace with destination directory path
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output directory
os.makedirs(destination_dir, exist_ok=True)

# Create train, val, and test directories in the destination folder
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(destination_dir, split)
    os.makedirs(split_dir, exist_ok=True)

# Move files into splits
for class_dir in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_dir)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        np.random.shuffle(images)

        train_split = int(train_ratio * len(images))
        val_split = int(val_ratio * len(images))

        # Creating subdirectories for each class in train, val, and test in the destination folder
        for split in ['train', 'val', 'test']:
            split_class_dir = os.path.join(destination_dir, split, class_dir)
            os.makedirs(split_class_dir, exist_ok=True)

        # Copy images to respective directories in the destination folder
        for i, image in enumerate(images):
            if i < train_split:
                split = 'train'
            elif i < train_split + val_split:
                split = 'val'
            else:
                split = 'test'

            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(destination_dir, split, class_dir, image)
            shutil.copy(src_path, dest_path)

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Define your transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = datasets.ImageFolder(root=os.path.join(destination_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(destination_dir, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(destination_dir, 'test'), transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

def get_model(pretrained=True):
    return {
        'EfficientNetV2': models.efficientnet_v2_s(pretrained=pretrained),
        'ResNet50': models.resnet50(pretrained=pretrained),
        'VGG16': models.vgg16(pretrained=pretrained),
        'MobileNetV3': models.mobilenet_v3_large(pretrained=pretrained),
        'DenseNet121': models.densenet121(pretrained=pretrained)
    }

def modify_model_classes(model_dict, num_classes):
    for name, model in model_dict.items():
        if name == 'EfficientNetV2':
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        elif name == 'ResNet50':
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif name == 'VGG16':
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        elif name == 'MobileNetV3':
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = torch.nn.Linear(num_ftrs, num_classes)
        elif name == 'DenseNet121':
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, num_classes)
    return model_dict

pretrained_models = modify_model_classes(get_model(pretrained=True), num_classes=101)
non_pretrained_models = modify_model_classes(get_model(pretrained=False), num_classes=101)


def train_and_validate_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = GradScaler()

    best_val_accuracy = 0

    # Lists to store per epoch metrics
    epoch_train_losses = []
    epoch_train_accuracies = []
    epoch_val_losses = []
    epoch_val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total
        epoch_train_losses.append(train_loss / len(train_loader))
        epoch_train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total

        epoch_val_losses.append(val_loss / len(val_loader))
        epoch_val_accuracies.append(val_accuracy)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Val Accuracy: {val_accuracy}')

        # Update best model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = model.state_dict()

        scheduler.step()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_train_losses, epoch_train_accuracies, epoch_val_losses, epoch_val_accuracies

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)

    return precision, recall, f1, accuracy


def train_and_evaluate_models(model_dict, pretrained_status, epochs=10):
    results_df = pd.DataFrame(columns=['Model', 'Pretrained', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])

    metrics_dict = {}

    for name, model in model_dict.items():
        print(f"Training and validating {name} (Pretrained: {pretrained_status})")
        trained_model, train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate_model(model,
                                                                                                             train_loader,
                                                                                                             val_loader,
                                                                                                             epochs=epochs)

        # Store metrics for plotting
        metrics_dict[name] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }

        # Saving the trained model
        model_path = os.path.join(model_save_dir,
                                  f'{name}_{"pretrained" if pretrained_status else "non_pretrained"}.pt')
        torch.save(trained_model.state_dict(), model_path)
        print(f"Saved trained model at {model_path}")

        print(f"Evaluating {name} (Pretrained: {pretrained_status})")
        precision, recall, f1, accuracy = evaluate_model(trained_model, test_loader)
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}")
        results_df.loc[len(results_df.index)] = [name, pretrained_status, precision, recall, f1, accuracy]

    return results_df, metrics_dict
    
# Train and evaluate pretrained models
pretrained_results_df, pretrained_metrics = train_and_evaluate_models(pretrained_models, True)

current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f'results_pretrained_{current_timestamp}.csv'
pretrained_results_df.to_csv(filename)

# Train and evaluate non-pretrained models
non_pretrained_results_df, non_pretrained_metrics = train_and_evaluate_models(non_pretrained_models, False)

current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f'results_nonpretrained_{current_timestamp}.csv'
non_pretrained_results_df.to_csv(filename)

with open(f'pt_metrics_{current_timestamp}.pickle', 'wb') as handle:
    pickle.dump(pretrained_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'npt_metrics_{current_timestamp}.pickle', 'wb') as handle:
    pickle.dump(non_pretrained_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plotting
for model_name in pretrained_models.keys():
    pt_epochs_range = range(1, 10 + 1)
    npt_epochs_range = range(1, 20 + 1)

    plt.figure(figsize=(12, 6))

    # Plot for pretrained model
    plt.subplot(1, 2, 1)
    plt.plot(pt_epochs_range, pretrained_metrics[model_name]["train_losses"], color='blue', label='Training Loss')
    plt.plot(pt_epochs_range, pretrained_metrics[model_name]["val_losses"], color='red', label='Validation Loss')
    plt.plot(pt_epochs_range, pretrained_metrics[model_name]["train_accuracies"], color='orange', label='Training Accuracy')
    plt.plot(pt_epochs_range, pretrained_metrics[model_name]["val_accuracies"], color='green', label='Validation Accuracy')
    plt.title(f'{model_name} Pretrained')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    # Plot for non-pretrained model
    plt.subplot(1, 2, 2)
    plt.plot(npt_epochs_range, non_pretrained_metrics[model_name]["train_losses"], color='blue', label='Training Loss')
    plt.plot(npt_epochs_range, non_pretrained_metrics[model_name]["val_losses"], color='red', label='Validation Loss')
    plt.plot(npt_epochs_range, non_pretrained_metrics[model_name]["train_accuracies"], color='orange', label='Training Accuracy')
    plt.plot(npt_epochs_range, non_pretrained_metrics[model_name]["val_accuracies"], color='green', label='Validation Accuracy')
    plt.title(f'{model_name} Non-Pretrained')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()