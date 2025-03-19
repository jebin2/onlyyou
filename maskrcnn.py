import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
import gc

class PersonExtractor:
	def __init__(self, model_type="X_101_32x8d", confidence_threshold=0.5, min_width=50, min_height=100, device=None):
		"""
		Initialize the person extractor with the specified model.
		
		Args:
			model_type: Model architecture to use ('X_101_32x8d' or 'R_50')
			confidence_threshold: Detection confidence threshold (0.0-1.0)
			min_width: Minimum person width in pixels to extract
			min_height: Minimum person height in pixels to extract
			device: Device to run inference on ('cuda' or 'cpu')
		"""
		self.min_width = min_width
		self.min_height = min_height
		self.person_class_id = 0  # COCO dataset class ID for "person"
		self.predictor = self._load_model(model_type, confidence_threshold, device)
		
	def _load_model(self, model_type, confidence_threshold, device=None):
		"""Load and configure the Mask R-CNN model"""
		cfg = get_cfg()
		
		# Set model architecture based on type
		if model_type == "R_50":
			config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
		else:  # Default to X_101_32x8d
			config_file = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
			
		cfg.merge_from_file(model_zoo.get_config_file(config_file))
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # COCO dataset has 80 classes
		cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
		
		# Set device if specified
		if device:
			cfg.MODEL.DEVICE = device
			
		print(f"Loading model: {config_file}")
		print(f"Using device: {cfg.MODEL.DEVICE}")
		
		return DefaultPredictor(cfg)
	
	def extract_from_file(self, image_path, output_folder="static", output_prefix=None):
		"""Extract people from a single image file"""
		# Create output directory if it doesn't exist
		os.makedirs(output_folder, exist_ok=True)
		
		# Load the image
		image = cv2.imread(image_path)
		if image is None:
			print(f"Error: Could not load image {image_path}")
			return []
			
		# Get filename without extension for output naming
		if output_prefix is None:
			output_prefix = Path(image_path).stem
		
		# Process the image
		return self.extract_from_image(image, output_folder, output_prefix)
		
	def extract_from_image(self, image, output_folder="static", output_prefix="person"):
		"""Extract people from an already loaded image"""
		# Run inference
		with torch.no_grad():  # Disable gradient calculation for inference
			outputs = self.predictor(image)

		# Get detected instances
		# instances = outputs["instances"]
		instances = outputs["instances"][outputs["instances"].pred_classes == 0]  # Class 0 = person
		pred_classes = instances.pred_classes.cpu().numpy()
		pred_masks = instances.pred_masks.cpu().numpy()
		pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
		
		# Find person indices
		person_indices = [i for i, cls in enumerate(pred_classes) if cls == self.person_class_id]
		
		output_paths = []
		
		# Process each detected person with a progress bar
		# for i in tqdm(person_indices, desc="Extracting people", unit="person", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
		for i in person_indices:
			mask = pred_masks[i]
			bbox = pred_boxes[i].astype(int)

			# Crop to bounding box
			x1, y1, x2, y2 = bbox
			width = x2 - x1
			height = y2 - y1
			
			# Skip small detections
			if width < self.min_width or height < self.min_height:
				continue
				
			# Two different extraction methods
			# Method 1: Use the mask directly on the original image (preserves original pixels)
			person = np.zeros_like(image)
			mask_3d = np.stack([mask] * 3, axis=2)  # Convert to 3-channel mask
			person = np.where(mask_3d, image, person)  # Apply mask
			
			# Crop to bounding box
			person_cropped = person[y1:y2, x1:x2]
			
			# Save the extracted person
			output_path = f"{output_folder}/{output_prefix}_{len(output_paths)}.png"
			cv2.imwrite(output_path, person_cropped)
			output_paths.append(output_path)

		# if len(output_paths) > 0:
		# 	print(f"Extracted {len(output_paths)} people from image")

		return output_paths
	
	def process_directory(self, input_dir, output_dir="extracted_people", file_ext=None):
		"""Process all images in a directory"""
		input_path = Path(input_dir)
		output_path = Path(output_dir)
		output_path.mkdir(exist_ok=True, parents=True)
		
		# Get all image files
		if file_ext:
			image_files = list(input_path.glob(f"*.{file_ext}"))
		else:
			image_files = []
			for ext in ['jpg', 'jpeg', 'png', 'bmp']:
				image_files.extend(input_path.glob(f"*.{ext}"))
		
		print(f"Found {len(image_files)} images to process")
		
		# Process each image with overall progress bar
		results = {}
		for img_path in tqdm(image_files, desc="Extracting Humans", unit="image", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
			output_prefix = img_path.stem
			extracted_paths = self.extract_from_file(
				str(img_path), 
				output_folder=str(output_path), 
				output_prefix=output_prefix
			)
			results[str(img_path)] = extracted_paths
			
		return results

def extract_people(image_path, output_folder="static", model_type="X_101_32x8d", confidence_threshold=0.5, min_width=50, min_height=100):
    """Simplified function to extract people from a single image"""
    extractor = None
    try:
        # Force garbage collection before loading model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Automatically select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize extractor
        extractor = PersonExtractor(
            model_type=model_type,
            confidence_threshold=confidence_threshold,
            min_width=min_width,
            min_height=min_height,
            device=device
        )
        
        # Process the image
        return extractor.extract_from_file(image_path, output_folder)
    finally:
        # Explicitly delete the model
        if extractor:
            del extractor.predictor
            del extractor
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Batch processing function
def process_directory(input_dir, output_dir="unlabeled_data", model_type="X_101_32x8d", confidence_threshold=0.5, min_width=50, min_height=100):
	"""Process all images in a directory"""
	try:
		# Force garbage collection before loading model
		gc.collect()
		torch.cuda.empty_cache() if torch.cuda.is_available() else None
		
		# Automatically select device
		device = "cuda" if torch.cuda.is_available() else "cpu"
		
		# Initialize extractor
		extractor = PersonExtractor(
			model_type=model_type,
			confidence_threshold=confidence_threshold,
			min_width=min_width,
			min_height=min_height,
			device=device
		)
		
		# Process all images
		return extractor.process_directory(input_dir, output_dir)
	finally:
		# Clean up
		gc.collect()
		torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Usage example
if __name__ == "__main__":
	process_directory("unlabeled_data")