from itertools import groupby

BOXOBAN_MAPPING = {
    ' ': 'empty',
    '#': 'wall',
    '$': 'box',
    '.': 'goal',
    '@': 'player'
}

BOXOBAN_INVERSE_MAPPING = {
    'empty': ' ',
    'wall': '#',
    'box': '$',
    'goal': '.',
    'player': '@'
}

def encode_boxoban_text(level):
    # Remove the first line of each level, which just contains the level number
    level = level[level.find("\n")+1:].strip()

    lines = []
    for line in level.split("\n"):
        # Group consecutive characters together and map them to their identities
        line_text = ", ".join([f"{len(list(iter))} {BOXOBAN_MAPPING[char]}" for char, iter in groupby(line)])
        lines.append(line_text)

    level_text = "\n".join(lines)

    return level_text

def decode_boxoban_text(text):
    # TODO: this code doesn't handle any error cases, which are sure to come up during generation
    level = ""

    for line in text.split("\n"):
        try:
            for section in line.split(", "):
                count, char = section.split(" ")
                level += BOXOBAN_INVERSE_MAPPING[char] * int(count)
        
            level += "\n"

        except:
            level += "Invalid line\n"

    return level

if __name__ == "__main__":
    level = """
##########
#   ######
# $     ##
#       ##
######   #
#### @ . #
## #.$# ##
# .$ .$  #
#       ##
##########
    """

    output = """10 wall
1 wall, 2 empty, 7 wall
1 wall, 1 empty, 1 box, 7 wall
1 wall, 1 goal, 1 empty, 7 wall
1 wall, 1 empty, 1 goal, 7 wall
1 wall, 1 empty, 1 box, 7 wall
1 wall, 1 empty, 1 goal, 7 wall
1 wall, 1 box, 1 empty, 1 box, 1 empty, 5 wall
1 wall, 1 player, 1 empty, 1 goal, 2 empty, 4 wall
10 wall"""

    print(decode_boxoban_text(output))

    # print(level)
    print(encode_boxoban_text(level))
    # print(decode_boxoban_text(encode_boxoban_text(level)))