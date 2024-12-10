from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration
import os

# LANGFUSE_SECRET_KEY="sk-lf-735a4295-7279-4e15-91bc-55e4beae235a"
# LANGFUSE_PUBLIC_KEY="pk-lf-6ec74791-6eea-46b0-9520-95b1bcadbfcb"
# LANGFUSE_HOST="https://langfuse.sboynton.io" # 🇪🇺 EU region

# os.environ['LANGFUSE_PUBLIC_KEY'] = LANGFUSE_PUBLIC_KEY
# os.environ['LANGFUSE_SECRET_KEY'] = LANGFUSE_SECRET_KEY
# os.environ['LANGFUSE_HOST'] = LANGFUSE_HOST
 
# @observe()
def story():
    openai.base_url = "https://litellm.sboynton.io"
    openai.api_key = "sk-v4twmPCGpuqYDXNqCGAyrpRerAMRt3PzneBY8bPBVFlozcYTFJP9iAUr54ctr9En"
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        messages=[
          {"role": "system", "content": "You are a great storyteller."},
          {"role": "user", "content": "Once upon a time in a galaxy far, far away..."}
        ],
    ).choices[0].message.content
 
# @observe()
def main():
    return story()
 
main()