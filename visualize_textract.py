import json
from PIL import Image, ImageDraw, ImageOps

image_path = "D:\\Workspace\\BOL-OCR\\BOL_Processing\\backend\\public\\uploads\\be612cd3-0ed7-4c92-b6ed-ad2c4bd08810.jpeg"
json_path = "D:\\Workspace\\BOL-OCR\\BOL_Processing\\backend\\public\\responses\\be612cd3-0ed7-4c92-b6ed-ad2c4bd08810_res.json"
output_path = "outputs/visualized.jpg"

image = Image.open(image_path)
image = ImageOps.exif_transpose(image)
width, height = image.size
draw = ImageDraw.Draw(image)

with open(json_path, "r") as f:
    response = json.load(f)

for block in response["blocks"]:
    if block["BlockType"] in ["KEY_VALUE_SET"]:
        box = block["Geometry"]["BoundingBox"]
        left = width * box["Left"]
        top = height * box["Top"]
        w = width * box["Width"]
        h = height * box["Height"]
        draw.rectangle([left, top, left + w, top + h], outline="red", width=2)

image.save(output_path)
print("Saved:", output_path)