from gemiwrap import GeminiWrapper
import os
from google import genai
import json
import shutil
import time

def find_file_recursive(base_path, filename):
    """Recursively search for a file in the given base directory."""
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)  # Return the full path if found
    return None  # Return None if not found

def clasify_image(directory_path, dest_path="labeled_data"):

	schema = genai.types.Schema(
		type = genai.types.Type.OBJECT,
		required = ["character_name"],
		properties = {
			"character_name": genai.types.Schema(
				type = genai.types.Type.STRING,
			),
		},
	)
	system_prompt = f"""You are an image analysis assistant specialized in identifying characters from various media including anime, cartoons, movies, TV shows, and video games. Your task is to analyze the given image and identify which character is present in the image.

Character identification guidelines:
1. Identify the character based on your own knowledge of characters from all types of media
2. Look for distinctive features such as hair color, hairstyle, clothing, accessories, and facial features
3. Consider the art style and context of the image
4. Apply a strict confidence threshold: only return a character name if you are at least 90% confident in the identification
5. If multiple characters appear in the image, then return 'none'
6. If you cannot identify the character with 90% or higher confidence, then return 'none'
7. If the image is unclear or of poor quality, then return 'none'

Important formatting rules:
- Your response should be limited to ONLY the character name or 'none'
- Character names must contain only lowercase alphabetic letters and numbers (a-z, 0-9)
- Do not include any special characters, spaces, or punctuation in the character name
- Do not provide any explanations or additional text
"""

	geminiWrapper = GeminiWrapper(system_instruction=system_prompt, schema=schema)
	with os.scandir(directory_path) as entries:
		for entry in entries:
			if entry.is_file():
				try:
					existing_file_path = find_file_recursive(dest_path, entry.name)
					
					if existing_file_path:
						print(f"File '{entry.name}' already exists in '{existing_file_path}', skipping...")
						continue  # Skip processing if file already exists
					
					# Extract character name if the file is not found
					character_name = json.loads(geminiWrapper.send_message(file_path=entry.path)[0])["character_name"]
					dest = os.path.join(dest_path, character_name)
					os.makedirs(dest, exist_ok=True)
					
					dest_file_path = os.path.join(dest, entry.name)
					shutil.copyfile(entry.path, dest_file_path)
					time.sleep(5)
				except:
					pass