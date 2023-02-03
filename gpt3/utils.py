import numpy as np
from PIL import Image
import os


def image_to_string_map(image_path, tile_size, tile_mapping):
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Get the image size
    width, height = image.size

    # Initialize an empty list to store the string map
    string_map = []

    # Iterate through the tiles
    for y in range(0, height, tile_size):
        row = []
        for x in range(0, width, tile_size):
            # Get the tile image
            tile = image.crop((x, y, x + tile_size, y + tile_size))

            # Get the average color of the tile
            avg_color = tile.getcolors(tile_size * tile_size)[0][1]

            # Look up the corresponding string in the mapping
            tile_string = tile_mapping.get(avg_color, "")

            # Append the tile string to the row
            row.append(tile_string)
        # Append the row to the string map
        string_map.append(row)

    return string_map


def string_to_list(string):
    # Split the string by newline characters
    string_list = string.split("\n")
    # Remove any empty strings from the list
    string_list = list(filter(None, string_list))
    return string_list



# Function to paste image tile on the level map
def paste_image_tile(level_map, tile_images, output_file):
    # Open the output file
    output_image = Image.new("RGBA", (len(level_map[0]), len(level_map)), (255, 255, 255, 255))

    # Iterate through the level map and paste the corresponding tile image
    for i in range(len(level_map)):
        for j in range(len(level_map[0])):
            char = level_map[i][j]
            tile_image = tile_images.get(char)
            if tile_image:
                output_image.paste(tile_image, (j, i))

    output_image.save(output_file)

    print(f"Level_size:{output_image.size[0]} x {output_image.size[1]}")


def count_tile(grid, tile="A"):
    count = 0
    for row in grid:
        count += row.count(tile)
    return count


def convert_imgstrmap_to_string(grid):
    return "\n".join(["".join(row) for row in grid])


def print_image_sizes_in_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            file_path = os.path.join(folder_path, file)
            with Image.open(file_path) as im:
                print(f'{file} - {im.size[0]}x{im.size[1]}')

