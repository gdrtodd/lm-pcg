import os
import openai
os.environ['OPENAI_API_KEY'] = "sk-..."
openai.api_key = os.getenv("OPENAI_API_KEY")

openai.File.create(
  file=open("example_data.jsonl", "rb"),
  purpose='fine-tune'
)