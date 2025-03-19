from frameGenerator import process
from maskrcnn import process_directory
from briaai import remove_background
from clasify_data_with_gemini import clasify_image

if __name__ == "__main__":
	process("dragon_ball_z_-_001.mp4")
	remove_background("unlabeled_data")
	process_directory("unlabeled_data")
	clasify_image("unlabeled_data")
