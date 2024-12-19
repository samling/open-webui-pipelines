from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import sys
import os
# sys.path.insert(0, os.path.abspath("./litellm"))
import litellm
import requests
from requests.models import PreparedRequest
from pprint import pformat
import json

# Request URL: https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent

# Request Headers: {
#   "Content-Type": "application/json",
#   "x-goog-api-key": "fake-key",
#   "user-agent": "google-genai-sdk/0.2.1 gl-python/3.11.9",
#   "x-goog-api-client": "google-genai-sdk/0.2.1 gl-python/3.11.9",
#   "Content-Length": "202"
# }

# Request Body: {
#   "contents": [
#     {
#       "parts": [
#         {
#           "text": "When is the next total solar eclipse in the United States?"
#         }
#       ],
#       "role": "user"
#     }
#   ],
#   "tools": [
#     {
#       "googleSearch": {}
#     }
#   ],
#   "generationConfig": {
#     "responseModalities": [
#       "TEXT"
#     ]
#   }
# }

# Monkey-patch the send method
original_send = requests.Session.send

def patched_send(self, request: PreparedRequest, **kwargs):
    print("\nRequest URL:", request.url)
    print("\nRequest Headers:", json.dumps(dict(request.headers), indent=2))
    if request.body:
        try:
            body = json.loads(request.body)
            print("\nRequest Body:", json.dumps(body, indent=2))
        except:
            print("\nRequest Body:", request.body)

    response = original_send(self, request, **kwargs)

    print("\nResponse:", response.text)

    return response

requests.Session.send = patched_send  # Apply the monkey patch


# client = genai.Client(
#     api_key="fake-key",
# )
model_id = "gemini/gemini-2.0-flash-exp"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

# response = client.models.generate_content(
#     model=model_id,
#     contents="When is the next total solar eclipse in the United States?",
#     config=GenerateContentConfig(
#         tools=[google_search_tool],
#         response_modalities=["TEXT"],
#     )
# )

tools = [{"googleSearch":{}}]
response = litellm.completion(
    model=model_id,
    messages=[{"role": "user", "content": "when is the next solar eclipse", "tool_calls": [{"googleSearch": {}}]}],
    tools=tools,
   
)
print(pformat(response.json()))

# for each in response.candidates[0].content.parts:
#     print(each.text)

# print(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)
