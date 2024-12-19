from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

gemini_api_key = "my-key"
project_id = "gen-lang-client-0352636303"
location = "us-central1"
app_creds = "/home/sboynton/Documents/HomeLab/apps-internal/scripts/vertex_key.json"

client = genai.Client(
    api_key=gemini_api_key
)
model_id = "gemini-2.0-flash-exp"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

response = client.models.generate_content_stream(
    model=model_id,
    contents="Who do the Celtics play next?",
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )
)

# Stream the response chunks
for chunk in response:
    if chunk.text:
        print(chunk.text)
    if chunk.candidates[0].grounding_metadata and chunk.candidates[0].grounding_metadata is not None:
        for grounding_chunk in chunk.candidates[0].grounding_metadata.grounding_chunks:
            print(grounding_chunk)
print(response)
# Print grounding metadata after streaming (if needed)
if hasattr(chunk, 'grounding_metadata') and chunk.grounding_metadata:
    print("\nGrounding metadata:")
    print(chunk.candidates[0].grounding_metadata.search_entry_point.rendered_content)