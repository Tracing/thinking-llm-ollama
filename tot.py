import json
import re
import requests
import utilities

def generate_answers(parameters):
    answers = []
    for _ in range(int(parameters["tot_n"])):
        answers.append(utilities.call_model(parameters["ollama_url"], parameters["tot_model"], parameters["prompt"], parameters["tot_num_ctx"], parameters["tot_t"]))
    return answers

def get_rating_prompt(user_query, answer, rating_prompt):
    return """{}
User Query: {}
Answer: {}""".format(rating_prompt, user_query, answer)

def get_rating_extraction_prompt(evaluation_answer, rating_extraction_prompt):
    return """{}
Text: {}""".format(rating_extraction_prompt, evaluation_answer)

def get_planning_prompt(user_query, planning_prompt_template):
    return """{}
User Query: {}""".format(planning_prompt_template, user_query)

def get_augmented_prompt(augmented_prompt_template, user_query, plan):
    return """{}
User Query: {}
Plan: {}""".format(augmented_prompt_template, user_query, plan)

def generate_planning_answers(parameters):
    plans = []
    planning_prompt = get_planning_prompt(parameters["prompt"], parameters["tot_planning_prompt_template"])
    for _ in range(int(parameters["tot_plan_n"])):
        plans.append(utilities.call_model(parameters["ollama_url"], parameters["tot_model"], planning_prompt, parameters["tot_num_ctx"], parameters["tot_t"]))
    return plans

def get_rating(parameters, answer):
    prompt = get_rating_prompt(parameters["prompt"], answer, parameters["tot_rating_prompt"])
    rating_answer = utilities.call_model(parameters["ollama_url"], parameters["tot_model"], prompt, parameters["tot_rating_num_ctx"], parameters["tot_rating_t"])
    rating = extract_rating(parameters, rating_answer)
    return rating

def extract_rating(parameters, rating_answer):
    prompt = get_rating_extraction_prompt(rating_answer["response"], parameters["tot_rating_extraction_prompt"])

    response = utilities.call_model(parameters["ollama_url"], parameters["tot_model"], prompt, parameters["tot_rating_extraction_num_ctx"], parameters["tot_rating_extraction_t"])
    match = re.search("[1-9][0-9]?", response["response"])
    if match is None:
        rating = None
    else:
        rating = int(match.group(0))
        if rating < 0 or rating > 10:
            rating = -1
    return rating

def tot(parameters):
    if parameters["tot_should_plan"]:
        (best_plan, plans, plan_ratings) = _tot(parameters, lambda x: generate_planning_answers(x))
        parameters["prompt"] = get_augmented_prompt(parameters["tot_augmented_prompt_template"], parameters["prompt"], best_plan["response"])
    return _tot(parameters, generate_answers)

def _tot(parameters, generate_answers_f):
    answers = generate_answers_f(parameters)
    ratings = []
    max_rating = -1
    best_answer = answers[0]
    for answer in answers:
        rating = sum([get_rating(parameters, answer) for _ in range(parameters["tot_n_ratings"])]) / parameters["tot_n_ratings"]
        ratings.append(rating)
        if rating is not None and rating > max_rating:
            max_rating = rating
            best_answer = answer
    return (best_answer, answers, ratings)

