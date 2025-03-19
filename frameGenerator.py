import cv2
from pathlib import Path
import json
import gc
import re
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm

class VideoFrameExtractor:
	def __init__(self, video_path: str, output_dir: str, resize: tuple = None, threshold: float = 30.0, compression_format: str = 'jpg', compression_quality: int = 90, max_workers: int = None):
		self.video_path = Path(video_path)
		self.output_dir = Path(output_dir)
		self.resize = resize
		self.threshold = threshold  # Threshold for scene change detection
		self.compression_format = compression_format.lower()  # jpg or png
		self.compression_quality = compression_quality  # 0-100 for jpg, 0-9 for png
		self.max_workers = max_workers or os.cpu_count()

		if not self.video_path.exists():
			raise FileNotFoundError(f"Video file not found: {video_path}")

	def _save_frame(self, frame, frame_filename):
		"""Save a frame to disk with optional resizing."""
		if self.resize:
			frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)  # INTER_AREA is better for downsizing

		# Optimize based on compression format
		if self.compression_format == 'jpg':
			cv2.imwrite(str(frame_filename.with_suffix('.jpg')), 
						frame, [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality])
			return str(frame_filename.with_suffix('.jpg'))
		else:
			cv2.imwrite(str(frame_filename.with_suffix('.png')), 
						frame, [cv2.IMWRITE_PNG_COMPRESSION, min(9, self.compression_quality // 10)])
			return str(frame_filename.with_suffix('.png'))

	def _process_batch(self, timestamps_batch, start_idx, pbar=None):
		"""Process a batch of timestamps efficiently."""
		cap = cv2.VideoCapture(str(self.video_path))
		results = []
		
		for i, timestamp in enumerate(timestamps_batch):
			frame_idx = start_idx + i
			cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
			ret, frame = cap.read()
			if ret:
				frame_filename = self.output_dir / f"scene_{frame_idx}"
				saved_path = self._save_frame(frame, frame_filename)
				results.append(saved_path)
			
			# Update progress bar if provided
			if pbar is not None:
				pbar.update(1)
		
		cap.release()
		return results

	def extract_frames(self):
		"""Extract keyframes from detected scene changes using optimized batch processing."""
		self.output_dir.mkdir(parents=True, exist_ok=True)
		
		# Detect scenes first
		print("Detecting scenes...")
		scene_timestamps = self._detect_scenes()
		print(f"Found {len(scene_timestamps)} scenes")
		
		if not scene_timestamps:
			print("No scenes detected with current threshold.")
			return {"fps": 0, "total_scenes": 0, "scene_frame_paths": [], "resolution": "0x0"}
		
		# Get video info
		cap = cv2.VideoCapture(str(self.video_path))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		cap.release()
		
		# Split timestamps into batches for each worker
		batch_size = max(1, len(scene_timestamps) // (self.max_workers * 2))
		timestamp_batches = [
			scene_timestamps[i:i + batch_size] 
			for i in range(0, len(scene_timestamps), batch_size)
		]
		
		all_frames = []
		
		# Create a progress bar for the entire process
		with tqdm(total=len(scene_timestamps), desc="Extracting Frames", unit="frame", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
			
			# Using ThreadPoolExecutor for parallel batch processing
			with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
				futures = {
					executor.submit(self._process_batch, batch, i * batch_size, pbar): i
					for i, batch in enumerate(timestamp_batches)
				}
				
				for future in as_completed(futures):
					batch_results = future.result()
					all_frames.extend(batch_results)
		
		# Sort frames by scene number
		all_frames.sort(key=lambda x: int(re.search(r'scene_(\d+)', x).group(1)))
		
		details = {
			"fps": fps,
			"total_scenes": len(scene_timestamps),
			"scene_frame_paths": all_frames,
			"resolution": f"{width}x{height}"
		}

		with open(self.output_dir / "details.json", 'w') as f:
			json.dump(details, f, indent=2)

		return details

	def _detect_scenes(self):
		"""Use pySceneDetect to find scene changes or return all timestamps if no threshold is set."""
		video = open_video(str(self.video_path))
		scene_manager = SceneManager()

		if self.threshold is not None:
			# Use Scene Detection
			scene_manager.add_detector(ContentDetector(threshold=self.threshold))
			scene_manager.detect_scenes(video, show_progress=True)
			scene_list = scene_manager.get_scene_list()
			return [scene[0].get_seconds() for scene in scene_list]
		
		else:
			# Return all frame timestamps if no threshold is provided
			fps = video.get_fps()
			total_frames = video.get_frame_count()
			return [frame / fps for frame in range(total_frames)]


def process(video_path, output_path="unlabeled_data", resize=None, threshold=20.0, 
			compression_format='jpg', compression_quality=90, max_workers=None):
	"""Main function to process video and extract frames from scene changes.
	
	Args:
		video_path: Path to the video file
		output_path: Directory to save frames
		resize: Optional tuple (width, height) to resize frames
		threshold: Threshold for scene detection (lower = more scenes)
		compression_format: 'jpg' or 'png' (jpg is faster)
		compression_quality: 0-100 for jpg, 0-9 for png
		max_workers: Number of worker threads (default: CPU count)
	"""
	try:
		# Force garbage collection before starting
		gc.collect()
		
		extractor = VideoFrameExtractor(
			video_path=video_path,
			output_dir=output_path,
			resize=resize,
			threshold=threshold,
			compression_format=compression_format,
			compression_quality=compression_quality,
			max_workers=max_workers
		)
		
		print(f"Processing video: {video_path}")
		result = extractor.extract_frames()
		print(f"Completed processing. Extracted {result['total_scenes']} scenes.")
		return result
	except Exception as e:
		print(f"Error: {e}")
		return None
	finally:
		# Another garbage collection at the end
		gc.collect()

# Usage example
if __name__ == "__main__":
	process(
		'dragon_ball_z_-_001.mp4',
		threshold=20.0,  # Lower threshold = more scene detections
		compression_format='jpg',  # JPG is much faster than PNG
		compression_quality=90,	# Good quality with smaller file size
		max_workers=os.cpu_count() # Use all available CPU cores
	)