import openai
import os

os.environ['OPENAI_API_KEY'] = "sk-..."
openai.api_key = os.getenv("OPENAI_API_KEY")


prompt = "The size of the level is 10 x 10. There is a base tile \"#\" with the count 77 and a space tile \" \" with the count 14. There are three fixed number tiles: \"$\", \"@\" and \".\". The count of \"$\" is 4, \".\" is 4, and \"@\" is 1.->"


openai.Completion.create(
  model="ada:ft-personal:level-gen-1-2023-01-31-07-15-28", # The model will be changed to whatever the name is
  prompt=,
  max_tokens=120,
  temperature=1,
  stop = [". END"]
)