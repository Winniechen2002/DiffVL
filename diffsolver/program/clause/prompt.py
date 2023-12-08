import json
import os
from diffsolver.config import TranslatorConfig

FILEPATH = os.path.dirname(os.path.abspath(__file__))


def v3(data):
    dsl = data["DSL"]
    dsl_prefix = dsl["prefix"]
    dsl_items = dsl["Items"]

    dsl_items_text = []
    for item_name, item in dsl_items.items():
        item_prefix = item["prefix"]
        item_examples = item["examples"]
        item_examples_text = []
        for example in item_examples:
            example_lang = example.get("lang", "")
            example_prog = example.get("prog", "")
            example_text = f"{example_prog} # {example_lang}"
            item_examples_text.append(example_text)
        item_examples_text = "- " + "\n- ".join(item_examples_text)
        item_text = f"{item_prefix}\n{item_examples_text}\n"
        dsl_items_text.append(item_text)
    dsl_items_text = "\n".join(dsl_items_text)

    single_clause = "We can compose those functions to implement various functions. Below are examples:\n"
    single_clause += '\n'.join(["# {lang}\n {prog}\n".format(
        **i)for i in data['example_sentence']])

    # Extract the examples section
    examples = data["examples"]
    examples_text = [
        "Below are the examples of the scenes and the program to solve the scenes. We first describe the objects in the scene and the tool we use. Then we describe the task and the program that can help to solve the task."
    ]


    for idx, scene in enumerate(examples):
        # Extract the scenes section

        tool = scene["tool_lang"]
        objects = scene["objects"]
        input = scene['lang']
        clauess = '\n- '.join(scene["clauses"])

        examples_text.append(
            f"Objects: {objects}\nInput: {input}\nProgram:\n- {clauess}\n"
        )

    examples_text = '\n\n'.join(examples_text)
    return f"{dsl_prefix}\n\n{dsl_items_text}\n\n{single_clause}\n\n{examples_text}"


def extract_info(json_str, config: TranslatorConfig):
    # Load the JSON string into a Python dictionary
    data = json.loads(json_str)
    if config.version == "v3":
        return v3(data)

    # Extract the DSL section
    dsl = data["DSL"]
    dsl_prefix = dsl["prefix"]
    dsl_items = dsl["Items"]

    single_clause = "We can compose those functions to form a clause. Below are examples "
    single_clause = '\n'.join(["Lang: {lang}\nProg: {prog}\nExplanation: {explanation}".format(
        **i)for i in data['single_clause']])

    # Extract the examples section
    examples = data["examples"]
    examples_text = [
        "Below are the examples of the scenes and the program to solve the scenes. We first describe the scene, and the objects. Then we describe the clauses, which are the program to solve the scene."
    ]

    for example in examples:
        # Extract the scenes section
        scenes = example["scenes"]
        tool = scenes["tool"]
        scene_description = scenes["description"]
        scene_prefix = scenes["prefix"]

        # Extract the clauses section
        clauses = example["clauses"]
        clauses_text = []
        for clause in clauses:
            clause_lang = clause.get("lang", "")
            clause_prog = clause.get("prog", "")
            clause_explanation = clause.get("explanation", "")
            if config.explanation:
                clauses_text.append(
                    f"Lang: {clause_lang}\nExplanation: {clause_explanation}\nProg: {clause_prog}\n")
            else:
                clauses_text.append(
                    f"Lang: {clause_lang}\nProg: {clause_prog}\n")

        examples_text.append(
            f"Scene Description: {scene_description}\nObjects: {scene_prefix}\nClauses:\n{''.join(clauses_text)}")

    # Group the extracted information into plain text
    dsl_text = f"{dsl_prefix}\n\n"
    dsl_items_text = []
    for item_name, item in dsl_items.items():
        item_prefix = item["prefix"]
        item_examples = item["examples"]
        item_examples_text = []
        for example in item_examples:
            example_lang = example.get("lang", "")
            example_prog = example.get("prog", "")
            example_text = f"{example_prog} # {example_lang}"
            item_examples_text.append(example_text)
        item_examples_text = "\n".join(item_examples_text)
        item_text = f"{item_prefix}\n{item_examples_text}\n"
        dsl_items_text.append(item_text)
    dsl_items_text = "\n".join(dsl_items_text)
    examples_text = "\n\n".join(examples_text)

    if 'weight_choose' in data:
        weight_choose = "Here are the constant weights for the functions tkeep and tlast, they are used for different constraints in DSL. We hope that every weight and function have a suitable corresponding relationship. Below are examples "
        weight_choose = '\n'.join(["Lang: {lang}\nProg: {prog}\nExample: {example}".format(
            **i)for i in data['weight_choose']])
        weight_choose = '\n\n' + weight_choose 
    else:
        weight_choose = ""

    return f"{dsl_text}{dsl_items_text}\n\n{single_clause}\n\n{examples_text}{weight_choose}"


def get_prompt(task, config: TranslatorConfig):
    with open(os.path.join(FILEPATH, f"input_{config.version}.json"), "r") as f:
        json_str = f.read()
        dsl = extract_info(json_str, config)
        return dsl + "\n\n" + task



if __name__ == "__main__":
    Task = "I now have a box and sphere. I need to first lift the sphere up and put it above the box"

    with open("output.txt", "w") as f:
        f.write(get_prompt(Task, TranslatorConfig(version="v3", explanation=True)))

