from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import gc

class BackgroundRemover:
    def __init__(self, model_name='briaai/RMBG-2.0', device='cuda' if torch.cuda.is_available() else 'cpu', image_size=(1024, 1024)):
        """
        Initialize the BackgroundRemover with a pre-trained model.
        
        Args:
            model_name (str): HuggingFace model name for image segmentation
            device (str): Device to run the model on ('cuda' or 'cpu')
            image_size (tuple): Size to resize images to for processing
        """
        self.device = device
        self.image_size = image_size
        
        # Load the model
        self.model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
        if device == 'cuda':
            torch.set_float32_matmul_precision('high')
        self.model.to(device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def remove_background(self, image_path, output_path=None, crop=False, bg_color=(0, 0, 0)):
        """
        Remove background from an image and replace with specified color.
        
        Args:
            image_path (str or Path): Path to the input image
            output_path (str or Path, optional): Path to save the output image
            crop (bool): Whether to crop the image to the subject's bounding box
            bg_color (tuple): RGB color to use for background (default: black)
            
        Returns:
            PIL.Image: Processed image with new background
        """
        # Load and preprocess the image
        image = Image.open(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate the mask
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        mask_pil = transforms.ToPILImage()(pred)
        mask = mask_pil.resize(image.size)
        
        # First create RGBA image with transparency
        image_rgba = image.convert("RGBA")
        temp_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        temp_image.paste(image_rgba, mask=mask)
        
        # Crop if requested (before adding background)
        if crop:
            temp_image = self._crop_to_content(temp_image)
        
        # Create new image with black background
        bg_image = Image.new("RGB", temp_image.size, bg_color)
        bg_image.paste(temp_image, mask=temp_image.split()[3])  # Use alpha channel as mask
        
        # Save the result if output path is provided
        if output_path:
            if isinstance(output_path, Path):
                output_path = str(output_path)
                
            # Can save as original format now since we don't have transparency
            bg_image.save(output_path)
            
        return bg_image
    
    def _crop_to_content(self, image):
        """
        Crop the image to the bounding box of the non-transparent content.
        
        Args:
            image (PIL.Image): RGBA image to crop
            
        Returns:
            PIL.Image: Cropped image
        """
        # Convert to numpy array for easier manipulation
        img_array = np.array(image)
        
        # Find non-transparent pixels (alpha > 0)
        alpha_channel = img_array[:, :, 3]
        non_empty_columns = np.where(alpha_channel.max(axis=0) > 0)[0]
        non_empty_rows = np.where(alpha_channel.max(axis=1) > 0)[0]
        
        # Check if there are any non-transparent pixels
        if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
            # Get the bounding box coordinates
            crop_box = (
                non_empty_columns.min(),
                non_empty_rows.min(),
                non_empty_columns.max() + 1,
                non_empty_rows.max() + 1
            )
            
            # Crop the image
            return image.crop(crop_box)
        
        return image
    
    def cleanup(self):
        """
        Clean up resources used by the model.
        """
        # Move model to CPU first if it was on GPU
        if self.device == 'cuda':
            self.model.to('cpu')
        
        # Delete the model and clear CUDA cache
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        print("Model unloaded and resources cleaned up")

def remove_background(folder):
    """
    Process all images in a folder and remove backgrounds, replacing with black.
    
    Args:
        folder (str): Folder containing images to process
    """
    remover = BackgroundRemover()
    input_path = Path(folder)
    
    # Find all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_files.extend(input_path.glob(f"*.{ext}"))
        image_files.extend(input_path.glob(f"*.{ext.upper()}"))

    print(f"Found {len(image_files)} images to process")
    
    try:
        # Process each image with overall progress bar
        for img_path in tqdm(image_files, desc="Removing Background", unit="image", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            try:
                # Keep original file path and extension
                remover.remove_background(
                    image_path=img_path,
                    output_path=img_path,  # Save back to original path
                    crop=True,
                    bg_color=(0, 0, 0)  # Black background
                )
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    finally:
        # Clean up resources even if processing is interrupted
        remover.cleanup()


# Example usage
if __name__ == "__main__":
    remove_background("unlabeled_data")