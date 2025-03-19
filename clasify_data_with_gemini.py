from gemiwrap import GeminiWrapper
import os
from google import genai
import json
import shutil
import time

def clasify_image(directory_path, dest_path="labeled_data"):

	labels = [
		"goku",
		"vegeta",
		"bulma",
		"piccolo",
		"gohan",
		"krillin",
		"trunks",
		"goten",
		"frieza",
		"cell",
		"buu",
		"yamcha",
		"tien",
		"chiaotzu",
		"master_roshi",
		"kami",
		"mr_popo",
		"dende",
		"korin",
		"yajirobe",
		"chi-chi",
		"videl",
		"hercule",
		"android_16",
		"android_17",
		"android_18",
		"android_19",
		"android_20",
		"dr_gero",
		"king_kai",
		"supreme_kai",
		"kibito",
		"old_kai",
		"dabura",
		"babidi",
		"beerus",
		"whis",
		"champa",
		"vados",
		"jaco",
		"hit",
		"goku_black",
		"zamasu",
		"future_trunks",
		"mai",
		"pilaf",
		"shu",
		"emperor_pilaf",
		"raditz",
		"nappa",
		"bardock",
		"king_vegeta",
		"tarble",
		"broly",
		"paragus",
		"cabba",
		"caulifla",
		"kale",
		"frost",
		"botamo",
		"magetta",
		"jiren",
		"toppo",
		"dyspo",
		"ribrianne",
		"kefla",
		"grand_priest",
		"zeno",
		"future_zeno",
		"gine",
		"king_cold",
		"cooler",
		"sorbet",
		"tagoma",
		"ginyu",
		"jeice",
		"burter",
		"recoome",
		"guldo",
		"zarbon",
		"dodoria",
		"cui",
		"appule",
		"nail",
		"guru",
		"king_kai",
		"gregory",
		"bubbles",
		"launch",
		"ox-king",
		"grandpa_gohan",
		"mr_satan",
		"fortuneteller_baba",
		"pikkon",
		"olibu",
		"uub",
		"pan",
		"bulla",
		"marron",
		"zeno",
		"grand_minister",
		"belmod",
		"khai",
		"marcarita",
		"vermoud",
		"quitela",
		"cognac",
		"sidra",
		"mojito",
		"rumsshi",
		"cus",
		"gowasu",
	]

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
				character_name = json.loads(geminiWrapper.send_message(file_path=entry.path)[0])["character_name"]
				dest = f"{dest_path}/{character_name}"
				os.makedirs(dest, exist_ok=True)
				shutil.copyfile(entry.path, f'{dest}/{entry.name}')
				time.sleep(5)