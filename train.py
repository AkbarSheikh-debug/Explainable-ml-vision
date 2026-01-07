import os
import sys
import torch
from torch.utils.data import DataLoader
import argparse
from torchvision import datasets
from utils import ExplainableML
import json

def train_model(data_dir='data/hymenoptera_data', 
                model_name='resnext101',
                num_epochs=25,
                batch_size=32,
                learning_rate=0.001,
                optimizer_name='adam',
                weight_decay=0.0001,
                momentum=0.9,
                scheduler_type='steplr',
                scheduler_step_size=7,
                scheduler_gamma=0.1,
                feature_extract=True,
                num_workers=4,
                output_dir='models'):
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ExplainableML
    explainer = ExplainableML(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model, input_size = explainer.initialize_model(model_name=model_name, feature_extract=feature_extract)
    model = model.to(explainer.device)
    
    # Create data transforms
    data_transforms = explainer.create_data_transforms(input_size)
    
    # Create datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    # Create dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, 
                      shuffle=True, num_workers=num_workers)
        for x in ['train', 'val']
    }
    
    # Get class names
    class_names = image_datasets['train'].classes
    print(f"Class names: {class_names}")
    print(f"Training on {len(image_datasets['train'])} images")
    print(f"Validation on {len(image_datasets['val'])} images")
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create optimizer based on choice
    optimizer_name_lower = optimizer_name.lower()
    if optimizer_name_lower == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name_lower == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name_lower == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        # Default to Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create learning rate scheduler
    scheduler_type_lower = scheduler_type.lower()
    if scheduler_type_lower == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_type_lower == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type_lower == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    else:
        # Default to StepLR
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    # Train the model
    print(f"Starting training for {num_epochs} epochs...")
    model, train_losses, val_losses, train_accs, val_accs = explainer.train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs
    )
    
    # Save the model
    model_path = os.path.join(output_dir, f'{model_name}_best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'input_size': input_size,
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    return model_path, train_losses, val_losses, train_accs, val_accs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an image classification model')
    parser.add_argument('--data_dir', type=str, default='data/hymenoptera_data',
                        help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='resnext101',
                        choices=['resnext101', 'resnet50', 'efficientnet_b0'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--scheduler', type=str, default='steplr',
                        choices=['steplr', 'cosine', 'exponential'],
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler_step_size', type=int, default=7,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                        help='Gamma for StepLR/ExponentialLR scheduler')
    parser.add_argument('--feature_extract', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Whether to freeze pretrained weights (True/False)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained model')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        scheduler_type=args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        feature_extract=args.feature_extract,
        num_workers=args.num_workers,
        output_dir=args.output_dir
    )