import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def get_data_transforms(input_size: int = 224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        # These constants come from the statistics of the ImageNet dataset
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    if model_name == "densenet":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def freeze_backbone(model, model_name: str, num_unfreeze_layers: int = 1):
    for param in model.parameters():
        param.requires_grad = False
    if model_name == "densenet":
        # Dynamically get feature layers + classifier
        feature_layers = list(model.features.children())  # e.g., [denseblock1, denseblock2, denseblock3, denseblock4, norm5]
        all_layers = feature_layers + [model.classifier]
    elif model_name == "resnet":
        # Dynamically get layer1 to layer4 + fc
        all_layers = [getattr(model, f'layer{i}') for i in range(1, 5)] + [model.fc]
    else:
        raise ValueError(f"Unsupported model for freezing: {model_name}")
    
    # Reverse to get last layers first
    all_layers.reverse()
    # Unfreeze the first num_unfreeze_layers from the reversed list
    for i in range(min(num_unfreeze_layers, len(all_layers))):
        for param in all_layers[i].parameters():
            param.requires_grad = True


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp: bool = False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, threshold: float = 0.8, use_amp: bool = False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] > threshold).long()  # Threshold on class 1 probability
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, precision, recall, f1


def visualize_predictions(model, test_loader, class_names, output_dir, threshold, device, num_samples=5):
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
            image = images[0].to(device)
            true_label = labels[0].item()
            
            output = model(image.unsqueeze(0))
            probs = torch.softmax(output, dim=1)[0]
            pred_label = int(probs[1] > threshold)
            conf = probs[1].item()
            
            # Denormalize for display
            img_display = image * std + mean
            img_display = img_display.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display * 255).astype('uint8')
            
            plt.figure(figsize=(6, 6))
            plt.imshow(img_display)
            plt.title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nConf: {conf:.2f}')
            plt.axis('off')
            plt.savefig(output_dir / f'sample_{i}.png', bbox_inches='tight')
            plt.close()


def build_dataloaders(dataset_dir: str, batch_size: int, seed: int = 42):
    train_transform, eval_transform = get_data_transforms()
    full_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
    class_names = full_dataset.classes
    if len(class_names) != 2:
        raise ValueError(f"Expected exactly 2 classes in dataset, found {len(class_names)}: {class_names}")

    # Get targets for stratified splitting
    targets = [label for _, label in full_dataset.samples]
    
    # Stratified split: 80% train, 10% val, 10% test
    train_indices, temp_indices, train_targets, temp_targets = train_test_split(
        range(len(full_dataset)), targets, test_size=0.2, stratify=targets, random_state=seed
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=temp_targets, random_state=seed
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)  # batch_size=1 for visualization
    return train_loader, val_loader, test_loader, class_names


def train_model(model_name: str, dataset_dir: str, output_path: str, epochs: int, batch_size: int, lr: float, device: torch.device, num_unfreeze_layers: int = 1, threshold: float = 0.8, weight_decay: float = 0.0, lr_factor: float = 0.1, lr_patience: int = 3, min_lr: float = 1e-6, mixed_precision: bool = False, early_stopping_patience: int = 20):
    print(f"Training {model_name} model on {dataset_dir} (unfreezing last {num_unfreeze_layers} layers, threshold={threshold}, weight_decay={weight_decay}, mixed_precision={mixed_precision})")
    train_loader, val_loader, _, class_names = build_dataloaders(dataset_dir, batch_size)
    model = get_model(model_name).to(device)
    freeze_backbone(model, model_name, num_unfreeze_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = torch.amp.GradScaler('cuda', enabled=mixed_precision)

    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, mixed_precision)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device, threshold, mixed_precision)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_prec={val_prec:.4f}, val_rec={val_rec:.4f}, val_f1={val_f1:.4f}, lr={current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "threshold": threshold,
                "config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_unfreeze_layers": num_unfreeze_layers,
                    "weight_decay": weight_decay,
                    "lr_factor": lr_factor,
                    "lr_patience": lr_patience,
                    "min_lr": min_lr,
                    "mixed_precision": mixed_precision,
                    "early_stopping_patience": early_stopping_patience,
                },
            }, output_path)
            print(f"Saved best model to {output_path} (val_acc={best_acc:.4f})")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs with no improvement in val_loss.")
            break

    print(f"Finished training {model_name}. Best validation accuracy: {best_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary classifiers with DenseNet121 and ResNet18.")
    parser.add_argument("--dataset-dir", type=str, default="./dataset", help="Path to dataset root with two class subfolders.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--model", type=str, choices=["densenet", "resnet", "all"], default="all", help="Which model(s) to train.")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save trained checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument("--unfreeze-layers", type=int, default=1, help="Number of final layers to unfreeze for fine-tuning (1=classifier only, 2+=more layers).")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for positive class prediction (0.0-1.0).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2 regularization) for optimizer.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor for ReduceLROnPlateau.")
    parser.add_argument("--lr-patience", type=int, default=10, help="Number of epochs with no improvement before reducing LR.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate after LR reduction.")
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="Number of epochs with no validation loss improvement before stopping.")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision training when using CUDA.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, class_names = build_dataloaders(args.dataset_dir, args.batch_size, args.seed)

    models_to_train = [args.model] if args.model in ["densenet", "resnet"] else ["densenet", "resnet"]
    for model_name in models_to_train:
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_output_dir / f"{model_name}_binary.pth"
        train_model(
            model_name=model_name,
            dataset_dir=args.dataset_dir,
            output_path=str(output_path),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            num_unfreeze_layers=args.unfreeze_layers,
            threshold=args.threshold,
            weight_decay=args.weight_decay,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            min_lr=args.min_lr,
            mixed_precision=args.mixed_precision,
            early_stopping_patience=args.early_stopping_patience,
        )
        
        # Load best model and evaluate on test set
        checkpoint = torch.load(output_path, map_location=device)
        model = get_model(model_name).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded model config: {checkpoint.get('config', 'No config saved')}")
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device, args.threshold, args.mixed_precision)
        print(f"{model_name} Test Results: loss={test_loss:.4f}, acc={test_acc:.4f}, prec={test_prec:.4f}, rec={test_rec:.4f}, f1={test_f1:.4f}")
        
        visualize_predictions(model, test_loader, class_names, model_output_dir, args.threshold, device, num_samples=5)
        print(f"Visualization saved to {model_output_dir}")


def load_model_for_inference(checkpoint_path: str, device: torch.device):
    """Load a trained model for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    model = get_model(model_name).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    class_names = checkpoint["class_names"]
    threshold = checkpoint["threshold"]
    config = checkpoint.get("config", {})
    return model, class_names, threshold, config


def predict_image(model, image_path: str, class_names: list, threshold: float, device: torch.device):
    """Predict class for a single image."""
    eval_transform = get_data_transforms()[1]  # eval transform
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image = eval_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = int(probs[1] > threshold)
        conf = probs[1].item()
    
    return class_names[pred], conf


if __name__ == "__main__":
    main()
