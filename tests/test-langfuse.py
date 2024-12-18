import uuid
from langfuse import Langfuse

def test_langfuse_trace():
    langfuse = Langfuse(
        secret_key="sk-fake-key",
        public_key="pk-fake-key",
        host="http://litellm.litellm:4000",
        debug=True,
    )
    test_body = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "chat_id": str(uuid.uuid4())
    }

    test_user = {
        "email": "my@email.com",
        "name": "Sam",
        "id": "sam-123"
    }
    trace = langfuse.trace(
        name="test-trace",
        input=test_body,
        user_id=test_user["email"],
        metadata={"user_name": test_user["name"], "user_id": test_user["id"]},
        session_id=test_body["chat_id"],
    )

    generation = trace.generation(
        name=test_body["chat_id"],
        model=test_body["model"],
        input=test_body["messages"],
    )


    trace_url = trace.get_trace_url()
    print(f"Test trace created: {trace_url}")

if __name__ == "__main__":
    test_langfuse_trace()
