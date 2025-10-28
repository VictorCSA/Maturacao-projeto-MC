import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from ripe_net import RipeNet
from dataset import FruitStateDataset
from backbone import create_backbone

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item()


def main(data_root, save_path, backbone_name, epochs, batch_size, lr, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformações mais robustas para treino (data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Transformações mais simples para validação
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Carregando datasets...")
    train_dataset = FruitStateDataset(root=data_root, split="train", transform=train_transform)
    val_dataset = FruitStateDataset(root=data_root, split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    n_fruits = len(train_dataset.fruit_classes)
    n_states = len(train_dataset.state_classes)
    print(f"Encontradas {n_fruits} classes de frutas: {train_dataset.fruit_classes}")
    print(f"Encontradas {n_states} classes de estados: {train_dataset.state_classes}")

    print("Criando o modelo RipeNet...")
    backbone = create_backbone(backbone_name)
    model = RipeNet(backbone=backbone, n_fruits=n_fruits, n_states=n_states)
    model.to(device)


    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print("-" * 20)
        print(f"Época {epoch + 1}/{epochs}")

        # --- Fase de Treinamento ---
        model.train()
        running_loss = 0.0
        correct_fruits_train = 0
        correct_states_train = 0
        
        for inputs, fruit_labels, state_labels in tqdm(train_loader, desc="Treinando"):
            inputs = inputs.to(device)
            fruit_labels = fruit_labels.to(device)
            state_labels = state_labels.to(device)

            optimizer.zero_grad()

            fruit_logits, state_logits = model(inputs)
            
            loss_fruit = criterion(fruit_logits, fruit_labels)
            loss_state = criterion(state_logits, state_labels)
            
            
            loss = loss_fruit + loss_state

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct_fruits_train += calculate_accuracy(fruit_logits, fruit_labels)
            correct_states_train += calculate_accuracy(state_logits, state_labels)
        
        train_loss = running_loss / len(train_dataset)
        train_acc_fruit = correct_fruits_train / len(train_dataset)
        train_acc_state = correct_states_train / len(train_dataset)

        print(f"Treino -> Loss: {train_loss:.4f} | Acurácia Fruta: {train_acc_fruit:.4f} | Acurácia Estado: {train_acc_state:.4f}")

        # --- Fase de Validação ---
        model.eval()
        running_loss_val = 0.0
        correct_fruits_val = 0
        correct_states_val = 0

        with torch.no_grad():
            for inputs, fruit_labels, state_labels in tqdm(val_loader, desc="Validando"):
                inputs = inputs.to(device)
                fruit_labels = fruit_labels.to(device)
                state_labels = state_labels.to(device)

                fruit_logits, state_logits = model(inputs)
                
                loss_fruit = criterion(fruit_logits, fruit_labels)
                loss_state = criterion(state_logits, state_labels)
                loss = loss_fruit + loss_state

                running_loss_val += loss.item() * inputs.size(0)
                correct_fruits_val += calculate_accuracy(fruit_logits, fruit_labels)
                correct_states_val += calculate_accuracy(state_logits, state_labels)

        val_loss = running_loss_val / len(val_dataset)
        val_acc_fruit = correct_fruits_val / len(val_dataset)
        val_acc_state = correct_states_val / len(val_dataset)
        
        # Acurácia combinada para salvar o melhor modelo
        val_acc_combined = (val_acc_fruit + val_acc_state) / 2

        print(f"Validação -> Loss: {val_loss:.4f} | Acurácia Fruta: {val_acc_fruit:.4f} | Acurácia Estado: {val_acc_state:.4f}")

        # Salvar o melhor modelo
        if val_acc_combined > best_val_acc:
            best_val_acc = val_acc_combined
            torch.save(model.state_dict(), save_path)
            print(f"Novo melhor modelo salvo em {save_path} com acurácia combinada de {best_val_acc:.4f}")

    print("Treinamento concluído!")

if __name__ == "__main__":
    data_root = r"Dataset"
    save_path = "trained.pth"
    backbone_name = "resnet18"
    epochs = 15
    batch_size = 128
    lr = 1e-4
    num_workers = 4
    
    main(data_root, save_path, backbone_name, epochs, batch_size, lr, num_workers)