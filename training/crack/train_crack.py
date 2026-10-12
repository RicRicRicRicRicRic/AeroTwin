import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add project root to sys.path to import MiniUNet directly from backend
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.app.models.base import MiniUNet
from backend.app.models.crack_detector import CRACK_CLASSES

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted([f.name for f in self.img_dir.iterdir() if f.suffix.lower() in valid_exts])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.img_dir / img_name
        
        mask_name = Path(img_name).stem + ".png"
        mask_path = self.mask_dir / mask_name
        if not mask_path.exists():
            mask_path = self.mask_dir / img_name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        
        # Convert mask to 2D LongTensor containing class indices (0 or 1)
        mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        mask_tensor = mask_transform(mask)
        mask = (mask_tensor > 0.5).long().squeeze(0)  # Shape: (512, 512)

        return image, mask

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = Path(__file__).parent / "dataset_cracks"
    
    train_img_dir = base_dir / "Train" / "images" if (base_dir / "Train").exists() else base_dir / "train" / "images"
    train_mask_dir = base_dir / "Train" / "masks" if (base_dir / "Train").exists() else base_dir / "train" / "masks"
    val_img_dir = base_dir / "val" / "images" if (base_dir / "val").exists() else base_dir / "Val" / "images"
    val_mask_dir = base_dir / "val" / "masks" if (base_dir / "val").exists() else base_dir / "Val" / "masks"

    img_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CrackDataset(train_img_dir, train_mask_dir, transform=img_transforms)
    val_dataset = CrackDataset(val_img_dir, val_mask_dir, transform=img_transforms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # Instantiate MiniUNet matching CRACK_CLASSES count (2 output channels)
    num_classes = len(CRACK_CLASSES)
    model = MiniUNet(in_channels=3, num_classes=num_classes).to(device)
    
    class_weights = torch.tensor([1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    num_epochs = 15
    best_val_loss = float('inf')
    
    output_weights_dir = PROJECT_ROOT / "backend" / "app" / "models" / "weights"
    output_weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_weights_dir / "crack_model.pt"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device, dtype=torch.long)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            print(f"  --> Saved new best weight state to {weights_path}")

    print("\nTraining completed successfully.")

if __name__ == "__main__":
    train()