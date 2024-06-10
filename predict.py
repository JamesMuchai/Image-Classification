import torch
from PIL import Image
import argparse
from tabulate import tabulate
import json
import numpy as np

def load_model(path):
    """
    Load a model checkpoint from the specified path.

    Args:
    - path (str): The path to the model checkpoint.

    Returns:
    - model (torch.nn.Module): The loaded model with its state dict.
    """
    checkpoint = torch.load(path)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_dict = checkpoint["class_dict"]
    return model

def parse_input():
    """
    Parse input arguments for the script.

    Returns:
    - argparse.Namespace: The parsed input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='Path to the image file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--category_names', type=str, help='Path to JSON dict with mapping of classes to category names.')
    parser.add_argument('--topk', type=int, default=1, help='Number of top categories to display.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference.')

    inputs = parser.parse_args()
    return inputs

def read_label_names(json_file):
    """
    Read label names from a JSON file.

    Args:
    - json_file (str): The path to the JSON file.

    Returns:
    - dict: The dictionary mapping classes to category names.
    """
    with open(json_file, 'r') as f:
        label_names = json.load(f)
    return label_names

def process_image(image_path):
    """
    Scale, crop, and normalize a PIL image for a PyTorch model.

    Args:
    - image_path (str): The path to the image file.

    Returns:
    - np.array: The processed image as a NumPy array.
    """
    norm_means = np.array([0.485, 0.456, 0.406])
    norm_sd = np.array([0.229, 0.224, 0.225])
    
    # Open the image
    image = Image.open(image_path)
    
    # Scale the image
    shortest_side = min(image.size)
    scale = 256 / shortest_side
    new_size = (int(image.width * scale), int(image.height * scale))
    im = image.resize(new_size)
    
    # Crop the image
    width, height = im.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    cropped_image = im.crop((left, top, right, bottom)).resize((224, 224))
    
    # Normalize the image
    np_image = np.array(cropped_image).astype(np.float32) / 255.0
    norm_image = (np_image - norm_means) / norm_sd
    
    # Transpose the image
    tr_img = np.transpose(norm_image, (2, 0, 1))
    
    return tr_img
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

def predict(image_path, model,device, topk=3):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    Args:
    - image_path (str): The path to the image file.
    - model (torch.nn.Module): The trained model.
    - topk (int): The number of top classes to return.

    Returns:
    - tuple: The top probabilities and corresponding classes.
    """
    
    model.to(device)
   
    # Process the image
    image_tensor = torch.from_numpy(process_image(image_path))
    image_tensor = image_tensor.to(torch.float32).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        prediction = model.forward(image_tensor)
        
    prob_score = torch.exp(prediction)
    top_p, top_class = prob_score.topk(topk, dim=1)
    
    return (top_p, top_class)

def format_output(probs, classes, json_dict, label_index):
    """
    Format and print the output predictions in a tabular format.

    Args:
    - probs (torch.Tensor): The probabilities of the top classes.
    - classes (torch.Tensor): The indices of the top classes.
    - json_dict (dict): The dictionary mapping class indices to category names.
    - label_index (dict): The dictionary mapping class indices to their labels.
    """
    keys = [label_index[int(i)] for i in classes[0]]
    labels = [json_dict[a] for a in keys]
    probabilities = [i * 100 for i in probs[0]]

    table = [["Class name", "Probability (%)"]]

    for i in range(len(probabilities)):
        row = [labels[i], probabilities[i]]
        table.append(row)

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))



if __name__ == '__main__':
    # Parse input arguments
    inputs = parse_input()
    # Check for GPU if requested
    if inputs.gpu:
        device = check_gpu()
    else: 
        device = torch.device("cpu")
    # Load label names from JSON file
    label_dict = read_label_names(inputs.category_names)
    
    # Load the model checkpoint
    model = load_model(inputs.checkpoint)
    
    # Map class indices to labels
    label_index = {v: k for k, v in model.class_to_idx.items()}
    
    # Predict the top classes
    probs, classes = predict(inputs.filename, model, device  inputs.topk)
    
    # Format and display the output
    format_output(probs, classes, label_dict, label_index)
