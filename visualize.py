"""
Generated with AI
Visualize images from the dataset using matplotlib.
"""
import matplotlib.pyplot as plt
from gan import DrawingSet
from typing import List

def visualize_dataset(file_names: List[str], num_images: int = 8, batch_size: int = 128):
    """
    Visualize images from the dataset.
    
    Args:
        file_names: List of .pt file paths to load
        num_images: Number of images to display
        batch_size: Batch size for the DataLoader
    """
    # Load dataset and create dataloader
    dataset = DrawingSet(file_names)
    dataloader = dataset.loader(batch_size=batch_size, shuffle=True)
    
    # Get first batch
    batch = next(iter(dataloader))
    
    # Take only num_images from the batch
    images = batch[:num_images]
    
    # Calculate grid dimensions
    ncols = 4
    nrows = (num_images + ncols - 1) // ncols
    
    # Create figure
    _, axes = plt.subplots(nrows, ncols, figsize=(12, 3*nrows)) # type: ignore
    axes = axes.flatten()
    
    # Display images
    for idx, img in enumerate(images):
        ax = axes[idx]
        
        # Handle channel dimension
        if img.shape[0] == 1:
            # Grayscale
            img_display = img.squeeze(0).cpu().detach().numpy()
            ax.imshow(img_display, cmap='gray')
        elif img.shape[0] == 3:
            # RGB
            img_display = img.cpu().detach().numpy().transpose(1, 2, 0)
            ax.imshow(img_display)
        else:
            ax.imshow(img.squeeze().cpu().detach().numpy(), cmap='gray')
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show() # type: ignore


if __name__ == "__main__":
    # Visualize dog and violin datasets
    visualize_dataset(["dog.pt", "violin.pt"], num_images=12)
