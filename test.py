import ollama

stream = ollama.chat(
    model="mistral",
    messages=[
        {"role": "user", "content": "What can you tell me about AI."}
    ],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)