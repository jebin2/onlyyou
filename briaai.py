import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def remove_background_preserve_position(self, image_path, output_path=None, save_alpha=True):
        """
        Remove background while preserving exact foreground position and size.
        
        Args:
            image_path (str or Path): Path to the input image
            output_path (str or Path, optional): Path to save the output image
            save_alpha (bool): If True, save with transparency (PNG RGBA), else black background
        
        Returns:
            PIL.Image: Processed image with background removed
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate the mask
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        mask_pil = transforms.ToPILImage()(pred)
        mask = mask_pil.resize(image.size, Image.LANCZOS)

        # Create result image preserving exact position and size
        if save_alpha:
            # Create RGBA image with transparency
            result_image = Image.new("RGBA", original_size, (0, 0, 0, 0))
            image_rgba = image.convert("RGBA")
            
            # Apply mask to create transparency
            mask_array = np.array(mask)
            image_array = np.array(image_rgba)
            
            # Set alpha channel based on mask
            image_array[:, :, 3] = mask_array
            
            result_image = Image.fromarray(image_array)
        else:
            # Create RGB image with black background
            result_image = Image.new("RGB", original_size, (0, 0, 0))
            # Paste original image using mask
            result_image.paste(image, mask=mask)

        # Save output if path provided
        if output_path:
            if isinstance(output_path, Path):
                output_path = str(output_path)
            
            if save_alpha:
                # Ensure PNG extension for transparency
                if not output_path.lower().endswith('.png'):
                    output_path = os.path.splitext(output_path)[0] + '.png'
                result_image.save(output_path, format="PNG")
            else:
                result_image.save(output_path)

        return result_image

    def remove_background(self, image_path, output_path=None, crop=False, bg_color=(0, 0, 0), save_alpha=False, keep_size=True, skip_crop=False):
        """
        Original method - kept for backward compatibility.
        For preserving exact position/size, use remove_background_preserve_position() instead.
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate the mask
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        mask_pil = transforms.ToPILImage()(pred)
        mask = mask_pil.resize(image.size)

        # Create RGBA image with transparency
        image_rgba = image.convert("RGBA")
        temp_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        temp_image.paste(image_rgba, mask=mask)

        # Skip cropping if requested
        if skip_crop:
            print(f"Skipping cropping for {image_path}")
        else:
            # Crop if requested
            if crop:
                temp_image = self._crop_to_content(temp_image)
                if keep_size:
                    # Paste cropped image back onto original canvas size
                    padded_image = Image.new("RGBA", original_size, (0, 0, 0, 0))
                    offset_x = (original_size[0] - temp_image.size[0]) // 2
                    offset_y = (original_size[1] - temp_image.size[1]) // 2
                    padded_image.paste(temp_image, (offset_x, offset_y))
                    temp_image = padded_image

        # Save output
        if output_path:
            if isinstance(output_path, Path):
                output_path = str(output_path)

            if save_alpha:
                if not output_path.lower().endswith('.png'):
                    output_path = os.path.splitext(output_path)[0] + '.png'
                temp_image.save(output_path, format="PNG")
            else:
                bg_image = Image.new("RGB", temp_image.size, bg_color)
                bg_image.paste(temp_image, mask=temp_image.split()[3])
                bg_image.save(output_path)

        return temp_image

    def _crop_to_content(self, image):
        """
        Crop the image to the bounding box of the non-transparent content.
        """
        img_array = np.array(image)
        alpha_channel = img_array[:, :, 3]
        non_empty_columns = np.where(alpha_channel.max(axis=0) > 0)[0]
        non_empty_rows = np.where(alpha_channel.max(axis=1) > 0)[0]

        if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
            crop_box = (
                non_empty_columns.min(),
                non_empty_rows.min(),
                non_empty_columns.max() + 1,
                non_empty_rows.max() + 1
            )
            return image.crop(crop_box)

        return image

    def cleanup(self):
        """
        Clean up resources used by the model.
        """
        if self.device == 'cuda':
            self.model.to('cpu')
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Model unloaded and resources cleaned up")


def remove_background_batch(folder, output_path=None, save_alpha=True, preserve_position=True):
    """
    Process all images in a folder and remove backgrounds while preserving position and size.

    Args:
        folder (str): Folder containing images to process
        output_path (str): Output folder path
        save_alpha (bool): If True, saves output as PNG with transparency
        preserve_position (bool): If True, keeps foreground in exact original position
    """
    remover = BackgroundRemover()
    input_path = Path(folder)
    
    # Create output directory if it doesn't exist
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
        image_files.extend(input_path.glob(f"*.{ext}"))
        image_files.extend(input_path.glob(f"*.{ext.upper()}"))

    print(f"Found {len(image_files)} images to process")

    try:
        for img_path in tqdm(image_files, desc="Removing Background", unit="image"):
            try:
                # Determine output filename
                if output_path:
                    output_filename = os.path.basename(img_path)
                    if save_alpha and not output_filename.lower().endswith('.png'):
                        output_filename = os.path.splitext(output_filename)[0] + '.png'
                    output_file = os.path.join(output_path, output_filename)
                else:
                    output_file = img_path

                if preserve_position:
                    # Use the new method that preserves exact position
                    remover.remove_background_preserve_position(
                        image_path=img_path,
                        output_path=output_file,
                        save_alpha=save_alpha
                    )
                else:
                    # Use original method with no cropping
                    remover.remove_background(
                        image_path=img_path,
                        output_path=output_file,
                        crop=False,  # No cropping to preserve position
                        bg_color=(0, 0, 0),
                        save_alpha=save_alpha,
                        skip_crop=True
                    )
                    
                # print(f"✓ Processed: {os.path.basename(img_path)}")
                
            except Exception as e:
                print(f"✗ Error processing {img_path}: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    finally:
        remover.cleanup()


# Single image processing function
def remove_background_single(image_path, output_path=None, save_alpha=True):
    """
    Process a single image and remove background while preserving position and size.
    
    Args:
        image_path (str): Path to input image
        output_path (str, optional): Path to save output image
        save_alpha (bool): If True, saves with transparency
    
    Returns:
        PIL.Image: Processed image
    """
    remover = BackgroundRemover()
    
    try:
        result = remover.remove_background_preserve_position(
            image_path=image_path,
            output_path=output_path,
            save_alpha=save_alpha
        )
        print(f"✓ Successfully processed: {os.path.basename(image_path)}")
        return result
    except Exception as e:
        print(f"✗ Error processing {image_path}: {str(e)}")
        return None
    finally:
        remover.cleanup()


# Example usage
if __name__ == "__main__":
    # Process entire folder - preserves exact position and size
    remove_background_batch(
        folder="../CaptionCreator/media/puzzle_x_pic/", 
        output_path="../CaptionCreator/media/processed/", 
        save_alpha=True,
        preserve_position=True
    )
    
    # Process single image
    # remove_background_single(
    #     image_path="path/to/your/image.jpg",
    #     output_path="path/to/output/image.png",
    #     save_alpha=True
    # )