# thinking-llm-ollama
An implementation of mixture of agents (https://arxiv.org/abs/2406.04692) algorithm for ollama.

Takes a prompt stored in prompt.txt (by default) and asks a group of llms to create a response to the prompt. An aggregator llm takes all of these responses as well as the user prompt and combines them to create a better response to the prompt. This is equalivent to a mixture of agents for one layer.

Set parameters in options.json
Modify prompt.txt to your query.
Run python3 main.py
The combined answer to the query should be in output.txt. (or wherever it is specified in options.json)

Requirements:
python3
ollama

Ollama server must be running and accessible.
