import os
import argparse
from PIL import Image, ImageOps
import rembg

def combine_original(input_folder):
    # 1. Validate folder
    if not os.path.isdir(input_folder):
        print(f"Error: The path '{input_folder}' is not a valid directory.")
        return

    # 2. Collect and sort PNG files
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.png')])

    if len(files) != 5:
        print(f"Error: Found {len(files)} PNGs, but exactly 5 are required for 'combine_original'.")
        return

    # 3. Process images (1024x1024)
    images = []
    for f in files:
        img = Image.open(os.path.join(input_folder, f)).convert("RGBA")
        # Ensure they are 1024x1024 before stitching
        if img.size != (1024, 1024):
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        images.append(img)

    # 4. Create a massive canvas for the 5 images (5120 x 1024)
    canvas = Image.new('RGBA', (5120, 1024))

    # 5. Paste 1024x1024 images side-by-side
    for i, img in enumerate(images):
        canvas.paste(img, (i * 1024, 0))

    # 6. Downscale the entire strip to 1024x256
    # This squeezes the aspect ratio as requested
    final_strip = canvas.resize((1280, 256), Image.Resampling.LANCZOS)

    # 7. Save in current working directory
    output_name = "output_original.png"
    final_strip.save(output_name)
    print(f"Successfully created {output_name} (downscaled to 1024x256) from: {input_folder}")
    
def combine_images(input_folder):
    # 1. Validate folder
    if not os.path.isdir(input_folder):
        print(f"Error: The path '{input_folder}' is not a valid directory.")
        return

    # 2. Collect and sort PNG files
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.png')])

    if len(files) != 4:
        print(f"Error: Found {len(files)} PNGs, but exactly 4 are required.")
        return

    # 3. Process images
    images = []
    for f in files:
        img = Image.open(os.path.join(input_folder, f)).convert("RGBA")
        # Ensure exact 256x256 size for consistent stitching
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
        images.append(img)

    # 4. Create canvas (1024 width x 256 height)
    canvas = Image.new('RGBA', (1024, 256))

    # 5. Paste side-by-side
    for i, img in enumerate(images):
        canvas.paste(img, (i * 256, 0))

    # 6. Save in current working directory
    output_name = "output.png"
    canvas.save(output_name)
    print(f"Successfully created {output_name} from images in: {input_folder}")

def pad_and_remove_background(image: Image.Image, border_size: int, rembg_session) -> Image.Image:
    # overflow object need first padding before run segmentation
    padded_image = ImageOps.expand(image, border=border_size, fill="white")

    foreground_image = rembg.remove(padded_image, session=rembg_session)

    width, height = foreground_image.size
    crop_box = (border_size, border_size, width - border_size, height - border_size)
    cropped_image = foreground_image.crop(crop_box)

    return cropped_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine 4 PNGs from a folder into a 1024x256 strip.")
    parser.add_argument("folder", type=str, help="Path to the folder containing 4 PNG images.")
    
    args = parser.parse_args()
    #im = Image.open("./examples/POSTER/spot_orig/001.png").convert("RGBA")
    #border_size = int(im.size[0] * 0.05)
    #rembg_session = rembg.new_session()
    #img = pad_and_remove_background(im, border_size, rembg_session)
    #img.save("./examples/POSTER/spot_orig/001_s.png")
    combine_images(args.folder)
    #combine_original(args.folder)