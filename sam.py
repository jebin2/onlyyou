import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectCropper:
    """Class to handle object cropping from segmentation masks."""
    
    def __init__(self, output_dir: str = "cropped_objects", pad: int = 10, 
                 save_masked: bool = True):
        """
        Initialize the ObjectCropper.
        
        Args:
            output_dir: Directory to save cropped objects
            pad: Padding to add around objects
            save_masked: Whether to save masked version of objects
        """
        self.output_dir = output_dir
        self.pad = pad
        self.save_masked = save_masked
        os.makedirs(output_dir, exist_ok=True)
        
    def _get_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box coordinates from a mask."""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None
            
        return (
            max(0, np.min(y_indices) - self.pad),
            min(mask.shape[0], np.max(y_indices) + self.pad),
            max(0, np.min(x_indices) - self.pad),
            min(mask.shape[1], np.max(x_indices) + self.pad)
        )
    
    def _save_cropped_image(self, image: np.ndarray, mask: Dict[str, Any], 
                           index: int) -> None:
        """Save a single cropped object."""
        m = mask['segmentation']
        bbox = self._get_bbox(m)
        
        if not bbox:
            return
            
        y_min, y_max, x_min, x_max = bbox
        
        # Crop the image and mask
        cropped_img = image[y_min:y_max, x_min:x_max].copy()
        cropped_mask = m[y_min:y_max, x_min:x_max]
        
        # Save regular crop
        output_path = os.path.join(self.output_dir, f"object_{index:03d}.jpg")
        Image.fromarray(cropped_img).save(output_path)
        
        # Save masked version if requested
        if self.save_masked:
            masked_img = cropped_img.copy()
            masked_img[~cropped_mask] = [0, 0, 0]
            masked_output_path = os.path.join(self.output_dir, f"object_{index:03d}_masked.jpg")
            Image.fromarray(masked_img).save(masked_output_path)
    
    def crop_and_save_objects(self, image: np.ndarray, masks: List[Dict[str, Any]], 
                             max_workers: int = 4) -> None:
        """
        Crop and save objects from an image based on segmentation masks.
        
        Args:
            image: Input image
            masks: List of segmentation masks
            max_workers: Maximum number of parallel workers
        """
        # Sort masks by area (largest first)
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        logger.info(f"Cropping {len(sorted_masks)} objects to {self.output_dir}")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._save_cropped_image, image, mask, i)
                for i, mask in enumerate(sorted_masks)
            ]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
                
        logger.info(f"Finished cropping objects to {self.output_dir}")


class SAM2Processor:
    """Class to handle SAM2 processing."""
    
    def __init__(self, model_cfg: str, checkpoint: str, device: str = "cuda"):
        """
        Initialize the SAM2 processor.
        
        Args:
            model_cfg: Model configuration file path
            checkpoint: Model checkpoint file path
            device: Device to run inference on
        """
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        logger.info(f"Building SAM2 model from {checkpoint}")
        self.predictor = SAM2AutomaticMaskGenerator(
            build_sam2(model_cfg, checkpoint).to(device)
        )
    
    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks for an image."""
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.dtype):
            return self.predictor.generate(image)


def save_visualization(image: np.ndarray, output_path: str, figsize: Tuple[int, int] = (20, 20)) -> None:
    """Save a visualization of the image."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)  # Close the figure to free memory
    logger.info(f"Saved visualization to {output_path}")


def process_image(image_path: str, model_cfg: str, checkpoint: str, 
                 pad: int = 10, device: str = "cuda", save_masked: bool = True) -> None:
    """
    Process a single image with SAM2.
    
    Args:
        image_path: Path to the input image
        model_cfg: Path to model configuration file
        checkpoint: Path to model checkpoint file
        pad: Padding to add around objects
        device: Device to run inference on
        save_masked: Whether to save masked versions of cropped objects
    """
    # Get base filename without extension
    file_base, ext = os.path.splitext(image_path)
    
    # Create output directory
    output_dir = f"{file_base}_objects"
    
    # Load and prepare image
    logger.info(f"Processing image: {image_path}")
    image = np.array(Image.open(image_path).convert("RGB"))
    
    # Initialize processors
    sam_processor = SAM2Processor(model_cfg, checkpoint, device)
    cropper = ObjectCropper(output_dir=output_dir, pad=pad, save_masked=save_masked)
    
    # Generate masks
    masks = sam_processor.generate_masks(image)
    logger.info(f"Generated {len(masks)} masks")
    
    # Save visualization
    output_viz_path = f'{file_base}_sam2{ext}'
    save_visualization(image, output_viz_path)
    
    # Crop and save objects
    cropper.crop_and_save_objects(image, masks)


def batch_process_images(image_paths: List[str], model_cfg: str, checkpoint: str, 
                        pad: int = 10, device: str = "cuda", save_masked: bool = True) -> None:
    """
    Process multiple images with SAM2.
    
    Args:
        image_paths: List of paths to input images
        model_cfg: Path to model configuration file
        checkpoint: Path to model checkpoint file
        pad: Padding to add around objects
        device: Device to run inference on
        save_masked: Whether to save masked versions of cropped objects
    """
    for image_path in image_paths:
        process_image(image_path, model_cfg, checkpoint, pad, device, save_masked)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    checkpoint = "sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    image_path = "unlabeled_data/scene_25.jpg"  # Change to your image path
    
    # Process single image
    process_image(
        image_path=image_path,
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        pad=10,
        device="cuda",
        save_masked=True
    )
    
    # Example of batch processing multiple images
    # image_paths = ["image1.png", "image2.jpg", "image3.png"]
    # batch_process_images(image_paths, model_cfg, checkpoint)