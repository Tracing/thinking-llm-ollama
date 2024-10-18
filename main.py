import time
import moa
import tot
import utilities

def run_moa():
    options_file = "options.json"
    parameters = utilities.read_options(options_file)

    start = time.time()
    parameters["prompt"] = utilities.read_file(parameters["prompt_file"])
    parameters["aggregator_prompt_template"] = utilities.read_file(parameters["moa_aggregator_prompt_template_file"])
    (final_response, aggregator_prompt) = moa.moa(parameters)

    utilities.write_to_file(parameters["output_file"], final_response["response"])
    utilities.write_to_file(parameters["moa_aggregator_prompt_output_file"], aggregator_prompt)
    
    end = time.time()
    print("Aggregator model consumed {} tokens.".format(final_response["n_tokens_read"]))
    print("Combined response consumed {} tokens.".format(final_response["n_tokens_used"]))
    print("Answer written to {}".format(parameters["output_file"]))
    print("Took {:.3f} seconds".format(end - start))

def process_tot_output(parameters, responses, ratings):
    s = []
    for (response, rating) in zip(responses, ratings):
        s.append("Response: {}\n".format(response["response"]))
        s.append("Rating: {}\n".format(rating))
    s = "".join(s)

    utilities.write_to_file(parameters["tot_output_file"], s)

def run_tot():
    options_file = "options.json"
    parameters = utilities.read_options(options_file)

    start = time.time()
    parameters["prompt"] = utilities.read_file(parameters["prompt_file"])

    parameters["aggregator_prompt_template"] = utilities.read_file(parameters["moa_aggregator_prompt_template_file"])

    parameters["tot_rating_prompt"] = utilities.read_file(parameters["tot_rating_prompt_file"])
    parameters["tot_rating_extraction_prompt"] = utilities.read_file(parameters["tot_rating_extraction_prompt_file"])
    parameters["tot_planning_prompt_template"] = utilities.read_file(parameters["tot_planning_prompt_template_file"])
    parameters["tot_augmented_prompt_template"] = utilities.read_file(parameters["tot_augmented_prompt_template_file"])
    (final_response, responses, ratings) = tot.tot(parameters)

    utilities.write_to_file(parameters["output_file"], final_response["response"])
    process_tot_output(parameters, responses, ratings)
    
    end = time.time()
    print("Answer written to {}".format(parameters["output_file"]))
    print("Took {:.3f} seconds".format(end - start))

def main():
    options_file = "options.json"
    parameters = utilities.read_options(options_file)

    assert parameters["use_moa"] or parameters["use_tot"]
    assert not (parameters["use_moa"] and parameters["use_tot"])

    if parameters["use_moa"]:
        print("Running mixture of agents")
        run_moa()
    else:
        print("Running tree of thought")
        run_tot()

if __name__ == "__main__":
    main()
