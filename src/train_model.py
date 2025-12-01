import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data_processor import get_dataloaders
from src.strike_model import StrikeNet

def train_model(   
    metadata_path,
    output_dir="models",
    batch_size=64,
    num_epochs=50,
    learning_rate=0.001,
    sequence_length=15,
    use_attention=True,
    weight_decay=1e-4,
    early_stopping_patience=15
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True) #keep this output directory
    
    train_loader, validation_loader, classes = get_dataloaders(
        metadata_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_workers=8,
        pin_memory=True
    )
    
    print(f"Classes: {classes}")
    num_classes = len(classes)
    
    model = StrikeNet(
        num_classes=num_classes,
        sequence_length=sequence_length,
        use_attention=use_attention
    )
    model = model.to(device) #move model to gpu
    
    class_counts = {}
    for batch in train_loader:
        _, labels, _ = batch
        for label in labels:
            if label.item() not in class_counts:
                class_counts[label.item()] = 0
            class_counts[label.item()] += 1
    
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()] #balance classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    class_criterion = nn.CrossEntropyLoss(weight=class_weights) #weighted loss for imbalanced classes
    outcome_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    steps_per_epoch = len(train_loader)  
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.3, #warmup for 30% of training
        div_factor=10,
        final_div_factor=100
    )
    
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None #mixed precision training
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        class_correct = 0
        class_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, class_labels, outcome_labels in progress_bar:
            inputs = inputs.to(device)
            class_labels = class_labels.to(device)
            outcome_labels = outcome_labels.to(device)
            
            optimizer.zero_grad() #clear gradients
            
            if scaler: #use mixed precision if available
                with torch.cuda.amp.autocast():
                    class_outputs, outcome_outputs = model(inputs)
                    class_loss = class_criterion(class_outputs, class_labels)
                    outcome_loss = outcome_criterion(outcome_outputs, outcome_labels)
                    loss = class_loss + 0.5 * outcome_loss #weighted combined loss
                
                scaler.scale(loss).backward() #scale gradients to prevent underflow
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #prevent exploding gradients
                
                scaler.step(optimizer)
                scaler.update()
            else:
                class_outputs, outcome_outputs = model(inputs)
                class_loss = class_criterion(class_outputs, class_labels)
                outcome_loss = outcome_criterion(outcome_outputs, outcome_labels)
                loss = class_loss + 0.5 * outcome_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step() #adjust learning rate
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = class_outputs.max(1)
            class_total += class_labels.size(0)
            class_correct += predicted.eq(class_labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():0.4f}",
                'acc': f"{100. * class_correct / class_total:0.2f}%",
                'lr': f"{scheduler.get_last_lr()[0]:0.6f}"
            })
        
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100. * class_correct / class_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        model.eval() #switch to evaluation mode
        running_loss = 0.0
        class_correct = 0
        class_total = 0
        
        progress_bar = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad(): #disable gradient tracking for validation
            for inputs, class_labels, outcome_labels in progress_bar:
                inputs = inputs.to(device)
                class_labels = class_labels.to(device)
                outcome_labels = outcome_labels.to(device)
                
                class_outputs, outcome_outputs = model(inputs)
                class_loss = class_criterion(class_outputs, class_labels)
                outcome_loss = outcome_criterion(outcome_outputs, outcome_labels)
                loss = class_loss + 0.5 * outcome_loss
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = class_outputs.max(1)
                class_total += class_labels.size(0)
                class_correct += predicted.eq(class_labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():0.4f}",
                    'acc': f"{100. * class_correct / class_total:0.2f}%"
                })
        
        validation_loss = running_loss / len(validation_loader.dataset)
        validation_accuracy = 100. * class_correct / class_total
        
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {train_loss:0.4f}, Acc: {train_accuracy:0.2f}%")
        print(f"  Validation Loss: {validation_loss:0.4f}, Acc: {validation_accuracy:0.2f}%")
        
        if validation_accuracy > best_validation_accuracy: #save best model
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': validation_accuracy,
                'epoch': epoch,
                'classes': classes
            }
            
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f"  Saved new best model with validation accuracy: {validation_accuracy:0.2f}%")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= early_stopping_patience: #early stopping
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(validation_losses, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(validation_accuracies, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png')) #save training plots
    
    print(f"Training complete. Best validation accuracy: {best_validation_accuracy:0.2f}%")
    return os.path.join(output_dir, 'best_model.pth')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StrikeNet model")
    parser.add_argument('--metadata', type=str, required=True, help='Path to sequences metadata CSV')
    parser.add_argument('--output', type=str, default='models', help='Output directory for model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=15, help='Sequence length')
    parser.add_argument('--no_attention', action='store_true', help='Disable temporal attention')
    
    args = parser.parse_args()
    
    train_model(
        metadata_path=args.metadata,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        sequence_length=args.seq_len,
        use_attention=not args.no_attention
    )