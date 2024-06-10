
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import time
from PIL import Image
import argparse

# Define the available model names and load pretrained models
MODEL_NAMES = ["vgg", "densenet"]
models_ = {
    "vgg": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
    "densenet": models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
}

# Define transformations for training and testing datasets
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def check_gpu():
    """
    Check for available GPU and return the appropriate device.
    
    Returns:
    - torch.device: The device to use (cuda, mps, or cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def check_classes(folder):
    """
    Count the number of unique classes in the dataset.

    Args:
    - folder (str): The path to the dataset folder.

    Returns:
    - int: The number of unique classes.
    """
    data = load_data(folder)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    
    unique_classes = set()
    for _, labels in dataloader:
        unique_classes.update(labels.numpy())
    
    print("Counting number of classes....")
    return len(unique_classes)

def load_model(model_name, hidden, categories):
    """
    Load a pretrained model and modify its classifier.

    Args:
    - model_name (str): The name of the model (vgg or densenet).
    - hidden (int): The number of hidden units in the new classifier.
    - categories (int): The number of output categories.

    Returns:
    - model (torchvision.models): The modified model.
    """
    print("Updating model....")
    model = models_[model_name]
    if model_name == "vgg":
        input_features = 25088
    else:
        input_features = 1024
    
    new_classifier = nn.Sequential(
        nn.Linear(input_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, categories),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = new_classifier
    return model

def parse_input():
    """
    Parse input arguments for the script.

    Returns:
    - argparse.Namespace: The parsed input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Path to the image folder. Should contain two subfolders labeled "train" and "test".')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Path to the folder for saving model checkpoints.')
    parser.add_argument('--arch', type=str, default='vgg', choices=MODEL_NAMES, help='CNN model architecture. Choose "vgg" or "densenet".')
    parser.add_argument('--learning_rate', type=float, default=0.007, help='Learning rate for model training optimizer.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    inputs = parser.parse_args()
    return inputs

def load_data(image_folder, testing=False):
    """
    Load the dataset with the appropriate transformations.

    Args:
    - image_folder (str): The path to the image folder.
    - testing (bool): Whether to use testing transformations.

    Returns:
    - datasets.ImageFolder: The loaded dataset.
    """
    if not testing:
        data = datasets.ImageFolder(image_folder, transform=TRAIN_TRANSFORMS)
    else:
        data = datasets.ImageFolder(image_folder, transform=TEST_TRANSFORMS)
    return data

def validation_loop(model, criterion, valid_folder):
    """
    Perform a validation loop and print accuracy and loss.

    Args:
    - model (torchvision.models): The model to validate.
    - criterion (nn.Module): The loss function.
    - valid_folder (str): The path to the validation dataset folder.
    """
    valid_data = load_data(valid_folder, testing=True)
    validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    model.eval()
    with torch.no_grad():
        for data_ in validation_loader:
            images, labels = data_
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            prob_score = torch.exp(output)
            loss = criterion(output, labels).item()
            
            top_p, top_class = prob_score.topk(1, dim=1)
            matches = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(matches.type(torch.FloatTensor))

        print(f'Accuracy: {accuracy.item() * 100:.2f}%')
        print(f'Validation loss: {loss / len(validation_loader):.3f}')

def training_loop(model, epochs, optimizer, criterion, train_folder, valid_folder, device):
    """
    Perform the training loop for the model.

    Args:
    - model (torchvision.models): The model to train.
    - epochs (int): The number of epochs.
    - optimizer (torch.optim.Optimizer): The optimizer for training.
    - criterion (nn.Module): The loss function.
    - train_folder (str): The path to the training dataset folder.
    - valid_folder (str): The path to the validation dataset folder.
    - device (torch.device): The device to use for training.
    """
    data = load_data(train_folder)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    
    model.to(device)
    epoch_count = 1
    for epoch in range(epochs):
        training_loss = 0
        epoch_start_time = time.time()
        
        for images, labels in dataloader:
            model.train()  # Set model to training mode
            images, labels = images.to(device), labels.to(device)  # Move images and labels to GPU
            
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
        
        epoch_end_time = time.time()
        epoch_loss = training_loss / len(dataloader)
        print(f"Training loss, epoch {epoch_count}: {epoch_loss:.3f}")
        print(f"Elapsed time: {epoch_end_time - epoch_start_time:.2f} seconds")
        epoch_count += 1

        if epoch_count % 3 == 0:
            validation_loop(model=model, criterion=criterion, valid_folder=valid_folder)

def train_model(input_args):
    """
    Train the model based on input arguments.

    Args:
    - input_args (argparse.Namespace): The parsed input arguments.
    """
    # Get variables from input args
    train_folder = f'{input_args.dir}/train'
    valid_folder = f'{input_args.dir}/test'
    epochs = input_args.epochs
    model_name = input_args.arch
    l_r = input_args.learning_rate
    hidden_units = input_args.hidden_units
    
    # Check number of unique categories
    categories = check_classes(train_folder)

    # Define model
    model = load_model(model_name, hidden_units, categories)

    # Define optimizer and criterion
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=l_r)

    # Check for GPU if requested
    if input_args.gpu:
        device = check_gpu()
    else: 
        device = torch.device("cpu")

    # Training loop
    training_loop(model=model, epochs=epochs, optimizer=optimizer, criterion=criterion, train_folder=train_folder, valid_folder=valid_folder, device=device)

    # Save model
    checkpoint = {
        "model": model,
        "state_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
        "class_dict":model.class_to_idx
    }
    file_name = f'{input_args.save_dir}/checkpoint.pth'
    torch.save(checkpoint, file_name)

if __name__ == '__main__':
    input_args = parse_input()
    train_model(input_args)
