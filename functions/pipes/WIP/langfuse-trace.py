from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration
import os

# os.environ['LANGFUSE_PUBLIC_KEY'] = LANGFUSE_PUBLIC_KEY
# os.environ['LANGFUSE_SECRET_KEY'] = LANGFUSE_SECRET_KEY
# os.environ['LANGFUSE_HOST'] = LANGFUSE_HOST
 
# @observe()
def story():
    openai.base_url = "http://litellm.litellm:4000"
    openai.api_key = "sk-fake-key"
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
