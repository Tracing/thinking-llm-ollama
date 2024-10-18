import utilities

def get_aggregator_prompt(aggregator_prompt_template, prompt, responses):
    final_aggregator_prompt = ["{}\n\nUser Query: {}\n\nResponses from models:\n\n".format(aggregator_prompt_template, prompt)]
    for (i, response) in enumerate(responses):
        final_aggregator_prompt.append("{}. {}\n\n".format(i+1, response["response"]))
    final_aggregator_prompt = "".join(final_aggregator_prompt)
    return final_aggregator_prompt

def moa(parameters):
    responses = [utilities.call_model(parameters["ollama_url"], model, parameters["prompt"], parameters["moa_num_ctx"], parameters["moa_t"]) for model in parameters["moa_models"]]
    aggregator_prompt = get_aggregator_prompt(parameters["aggregator_prompt_template"], parameters["prompt"], responses)
    final_response = utilities.call_model(parameters["ollama_url"], parameters["moa_aggregator_model"], aggregator_prompt, parameters["moa_num_ctx_aggregator"], parameters["moa_t_aggregator"])
    return (final_response, aggregator_prompt)
