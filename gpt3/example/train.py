import openai
import os


os.environ['OPENAI_API_KEY'] = "sk-..."
openai.api_key = os.getenv("OPENAI_API_KEY")


openai.FineTune.create(training_file="file-XGinujblHPwGLSztz8cPS8XY")



 