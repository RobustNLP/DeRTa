import copy
import json, torch
import random


def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as g:
        for each in data:
            line = json.dumps(each)
            g.write(line + '\n')
    print("\n\n### WRITE ### \n data has been saved to {}\n### WRITE ###\n\n".format(filename))


def read_from_json(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as g:
        lines = g.readlines()
        for line in lines:
            sample = json.loads(line)
            data.append(sample)
    print("\n\n### READ ### \n read data from json\n### READ ###\n\n")
    return data


overall_beaver = read_from_json("safety_beaver_safe_and_unsafe_response.json")

unsensor_data = []
for i in range(6):
    unsensor_data += read_from_json("helpfulness_unsencor_wizardlm_60k_part_{}.json".format(i))
# unsensor_data = read_from_json("helpfulness_unsencor_wizardlm_60k.json")
final_unsensor_data = []
for each in unsensor_data:
    each["safe"] = "false"
    final_unsensor_data.append(
        {"prefix": "[INST] {} [\INST]".format(each["instruction"]), "text": " {}<|end_of_text|>".format(each["output"]),
         "safe": "false"})

Sorry_data = []  # item: {"instruction": <harmful question>, "response": <harmful response>, "safe": True}
# "safe": True -> means these data are used to refusal token prediction
# "safe": False -> means these data are used to next token prediction

Concate_refuse_data = []  # item: {"instruction": <harmful question>, "response": <partial harmful response> + <safe response>, "safe": False}
Vanilla_safety_data = []  # item: {"instruction": <harmful question>, "response": <safe response>, "safe": False}

for i, each in enumerate(overall_beaver):
    Sorry_data.append(
        {"prefix": "[INST] {} [\INST]".format(each["instruction"]), "text": " " + each["output"] + "<|end_of_text|>",
         "safe": "true"})

    words = each["output"].split(" ")
    length = len(words)
    try:
        ids = random.randint(1, length - 2)
    except:
        continue

    if i % 3 == 0:
        ids = 0

    res = " " + each["safe_response"] + "<|end_of_text|>"

    prefix = "[INST] {} [\INST]".format(each["instruction"]) + " " + " ".join(words[:ids])
    Concate_refuse_data.append({"prefix": prefix, "text": res, "safe": "false"})

    prefix = "[INST] {} [\INST]".format(each["instruction"])
    Vanilla_safety_data.append({"prefix": prefix, "text": res, "safe": "false"})

saved_data = final_unsensor_data + Concate_refuse_data[:6000]
write_to_json(saved_data, "llama_recaug.json")

saved_data = final_unsensor_data + Sorry_data[:3000] + Concate_refuse_data[:3000]
write_to_json(saved_data, "llama_derta.json")

saved_data = final_unsensor_data + Vanilla_safety_data[:6000]
write_to_json(saved_data, "llama_vanilla.json")
