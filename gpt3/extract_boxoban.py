import json
import pandas as pd



def extract_patterns_to_list(file_name):

    """
    A function that takes boxoban style data and converts it into List[levels: str].
    Each str is a level.
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()

    patterns = []
    pattern = []
    pattern_start = False
    for line in lines:
        if line.startswith(';'):
            pattern_start = False
            if pattern:
                patterns.append(''.join(pattern))
                pattern = []
        if line == '##########\n':
            pattern_start = True
        if pattern_start:
            pattern.append(line.replace(" ", "-"))

    return patterns[:-2]

def extract_pattern_info(pattern):
    """
    A function that takes in a str of level and returns a prompt.
    """
    lines = pattern.split('\n')
    len_of_rows = len(lines[0])
    no_of_rows = len(lines) -2 # Subtracting 2 '\n's at the end of last row. Case specific hard coding :/
    #print(len_of_rows,no_of_rows)
    no_of_base_tiles = pattern.count('#')
    no_of_space_tiles = pattern.count('-')
    no_of_dollar_tiles = pattern.count('$')
    no_of_dot_tiles = pattern.count('.')
    no_of_at_tiles = pattern.count('@')

    return (
        f'The size of the level is {len_of_rows} x {no_of_rows}. There is a base tile "#" with the count {no_of_base_tiles} and a space tile " " with the count {no_of_space_tiles}. There are three fixed number tiles: "$", "@" and ".". The count of "$" is {no_of_dollar_tiles}, "." is {no_of_dot_tiles}, and "@" is {no_of_at_tiles}.'
    )

def extract_patterns_info(file_name):
    """
    A function that takes List[str] of levels and returns List[prompts: str]
    """
    patterns = extract_patterns_to_list(file_name)
    return [extract_pattern_info(pattern) for pattern in patterns]


def create_data(prompts, completions, prompt_type):
    """
    A function that takes in prompts and completions and generates a JSONL file
    that is required to fine-tune GPT-3.
    """
    data = []
    if prompt_type == "parser":
        for i, parser in enumerate(prompts):
            prompt = {"prompt": f"{parser}->", "completion": f" {completions[i][:-2]}. END"}
            data.append(prompt)

    elif prompt_type == "sampler":
        for i, parser in enumerate(prompts):
            prompt = {"prompt": "Map: ->", "completion": f" {completions[i][:-2]}. END"}
            data.append(prompt)

    with open("data_6000.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def get_data(file):
    prompts = extract_patterns_info(file)
    completions = extract_patterns_to_list(file)

    comps = []

    for completion in completions:
        comps.append(completion[:-2])

    data = create_data(prompts, comps,prompt_type="sampler")
    df = pd.DataFrame({'level':comps})
    df.to_csv("data_6000.csv")





get_data("data_6000.txt")