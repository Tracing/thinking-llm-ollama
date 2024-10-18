import json
import requests

def call_model(ollama_url, model, prompt, num_ctx, t):
    url = "{}/api/generate".format(ollama_url)
    parameters = {"model": model,
                  "prompt": prompt,
                  "stream": False,
                  "options": {
                    "num_ctx": num_ctx,
                    "temperature": t
                  }}
    parameters = json.dumps(parameters)
    response = requests.post(url, data=parameters)
    response = json.loads(response.content)
    n_tokens_read = response["prompt_eval_count"]
    n_tokens_used = response["prompt_eval_count"] + response["eval_count"]
    tps = response["eval_count"] / response["eval_duration"] * 10 ** 9
    message = response["response"]

    return {"response": message,
            "n_tokens_read": n_tokens_read,
            "n_tokens_used": n_tokens_used,
            "tps": tps}

def read_file(f_name):
    with open(f_name, "r") as f:
        s = f.read()
    return s

def write_to_file(f_name, s):
    with open(f_name, "w") as f:
        f.write(s)

def read_options(f_name):
    with open(f_name, "r") as f:
        options = json.load(f)
    return options
    