import json
import requests
import time

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

def get_aggregator_prompt(aggregator_prompt_template, prompt, responses):
    final_aggregator_prompt = ["{}\n\nUser Query: {}\n\nResponses from models:\n\n".format(aggregator_prompt_template, prompt)]
    for (i, response) in enumerate(responses):
        final_aggregator_prompt.append("{}. {}\n\n".format(i+1, response["response"]))
    final_aggregator_prompt = "".join(final_aggregator_prompt)
    return final_aggregator_prompt

def repeat_moa(parameters):
    responses = [call_model(parameters["ollama_url"], model, parameters["prompt"], parameters["num_ctx"], parameters["t"]) for model in parameters["models"]]
    aggregator_prompt = get_aggregator_prompt(parameters["aggregator_prompt_template"], parameters["prompt"], responses)
    final_response = call_model(parameters["ollama_url"], parameters["aggregator_model"], aggregator_prompt, parameters["num_ctx_aggregator"], parameters["t_aggregator"])
    return (final_response, aggregator_prompt)

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

def main():
    options_file = "options.json"
    parameters = read_options(options_file)

    start = time.time()
    parameters["prompt"] = read_file(parameters["prompt_file"])
    parameters["aggregator_prompt_template"] = read_file(parameters["aggregator_prompt_template_file"])
    (final_response, aggregator_prompt) = repeat_moa(parameters)

    write_to_file(parameters["output_file"], final_response["response"])
    write_to_file(parameters["aggregator_prompt_output_file"], aggregator_prompt)
    
    end = time.time()
    print("Aggregator model consumed {} tokens.".format(final_response["n_tokens_read"]))
    print("Combined response consumed {} tokens.".format(final_response["n_tokens_used"]))
    print("Answer written to {}".format(parameters["output_file"]))
    print("Took {:.3f} seconds".format(end - start))

if __name__ == "__main__":
    main()