import math
import multiprocessing
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import random
import numpy as np
from tqdm import tqdm
import json
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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


random.seed(0)


# Read input data for inference
def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
    return input_data


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

my_device_mp = {}


def load_prompts(create_prompt, eval_data):
    eval_dataset = read_from_json(eval_data)  # [-100:]

    if "mmlu" in eval_data:
        few_shot = "Question: If consumers are advised that multigrained bread will substantially lessen the risk of cancer, which of the following will happen in the market for multigrained bread?\n-options\n-(A) The demand curve will shift to the left, decreasing the price of multigrained bread.\n-(B) The supply curve will shift to the left, increasing the price of multigrained bread.\n-(C) The demand curve will shift to the right, increasing the price of multigrained bread.\n-(D) The supply curve will shift to the right, decreasing the price of multigrained bread.\nAnswer: C \n\n\nQuestion: The bond length between any two nonmetal atoms is achieved under which of the following conditions?\n-options\n-(A) Where the energy of interaction between the atoms is at its minimum value\n-(B) Where the nuclei of each atom exhibits the strongest attraction to the electrons of the other atom\n-(C) The point at which the attractive and repulsive forces between the two atoms are equal\n-(D) The closest point at which a valence electron from one atom can transfer to the other atom\nAnswer: A \n\n\nQuestion: What is the name of the baby who appears in cartoons and comic strips with Popeye the Sailor?\n-options\n-(A) Pun\'kin\n-(B) Lamikins\n-(C) Suga\'baby\n-(D) Swee\'pea\nAnswer: D"
        few_shot = "Choose the correct option for the given question.\n"
        # few_shot = "Please think step by step to choose the right option. First provide your analysis, at the end, output ###Final answer: {answer option}.\n"
        eval_dataset = [
            {"question": few_shot + "\n\n\nQuestion: {}\n".format(each["question"]), "answer": each["answer"]} for each
            in eval_dataset]
        ps = [{"question": each["question"]} for each in eval_dataset]
        ps = create_prompt(ps)
        res = [p + "---answer---" + a["answer"] for p, a in zip(ps, eval_dataset)]

    elif "gsm8k" in eval_data:
        few_shot = "Question: Patty's Plumbing charges $40 to visit a house to make a repair, plus $35 per hour, or part thereof, for labor, plus parts. One job took 2.25 hours and used $60 in parts. How much did Patty charge?\n\nAnswer: She charged for 3 hours.\nThe total cost from the hourly rate was 3*35=$<<3*35=105>>105.\nShe charged 40+105+60=<<40+105+60=205>>205.\n#### 205 \n\n\nQuestion: Bobby takes a 30 min lunch and 2 15 minutes break per day at the office.  After 5 days, how many hours do his lunches and breaks add up to?\nAnswer: He has 2 15 minutes breaks per day so together they add up to 2*15 = <<2*15=30>>30 minutes\nHe also has a 30-minute lunch every day so together with his 30-minute break he has a 30+30 = 60-minute break each day\nEvery day his breaks total 60 minutes and he takes these breaks 5 days per week so he takes 60*5 = <<60*5=300>>300 minutes of break time\n60 minutes is in an hour so 300/60 = <<300/60=5>>5 hours of break time\n#### 5 \n\n\nQuestion: Arnold, Madeline, Camden, and Sarah painted 56 easter eggs. Arnold and Madeline painted the same number of eggs. Camden and Sarah painted a total of 30 eggs, but Camden painted 12 more than Sarah. How many more eggs did Camden paint than Arnold?\nAnswer: Arnold and Madeline painted 56 - 30 = <<56-30=26>>26 easter eggs.\nArnold and Madeline each painted 26/2 = <<26/2=13>>13 easter eggs.\nSarah painted (30 - 12)/2 = 18/2 = <<(30-12)/2=9>>9 easter eggs.\nCamden painted 9 + 12 = <<9+12=21>>21 easter eggs.\nCamden painted 21 - 13 = <<21-13=8>>8 eggs more than Arnold.\n#### 8"
        few_shot = ""
        # few_shot = "Please think step by step.\n"
        eval_dataset = [{"question": few_shot + "\n {} \n ".format(each["question"]), "answer": each["answer"]} for each
                        in eval_dataset]

        ps = [{"question": each["question"]} for each in eval_dataset]
        ps = create_prompt(ps)
        res = [p + "---answer---" + a["answer"] for p, a in zip(ps, eval_dataset)]
    else:
        res = create_prompt(eval_dataset)

    return res


def train_gpu(rank, args, world_size, gpus_per_process):
    if args.mode == "completion_attack":
        def create_prompt(instruct):
            final_instructions = []
            for example in instruct:
                final_input = example
                final_instructions.append(final_input)
            sources = [
                "{}".format(
                    example["question"]) for example in final_instructions]  # \n\n### Response:
            return sources
    else:
        def create_prompt(instruct):
            final_instructions = []
            for example in instruct:
                final_input = example
                final_instructions.append(final_input)
            sources = [
                "[INST] \n{}\n\n [\INST]".format(
                    example["question"]) for example in final_instructions]  # \n\n### Response:
            return sources

    results = []

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank)

    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:23457',
        world_size=world_size,
        rank=rank
    )

    torch.manual_seed(0)

    model_name_or_path = args.model_name_or_path

    output_file = args.saved_file_name_or_path
    output_file = output_file

    prompt = load_prompts(create_prompt, args.eval_data)

    to_use_fast = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=args.block_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.pad_token_id,
        # temperature=temperature
    )

    # Generate

    dataset = prompt
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch, shuffle=False, )

    # print("{} gpu inference start".format(device_ids))
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")

    for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", ascii=True, ncols=100)):
        answers = None
        if "---answer---" in batch[0]:
            new_batch = []
            answers = []
            for each in batch:
                es = each.split("---answer---")
                question = es[0]
                answer = es[1]
                new_batch.append(question)
                answers.append(answer)
            batch = new_batch

        model.eval()
        print('\n\n')
        with torch.no_grad():
            tokenized = tokenizer(batch, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.cuda()
            attn_mask = tokenized.attention_mask.cuda()
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

            try:
                with torch.no_grad():
                    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask,
                                                   generation_config=gen_config)
                decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            except Exception as e:
                print(e)
                print("Something wrong\n\n\n\n" * 3)
                continue
            print([ decoded_tokens[0].replace("### Response:", "\n\n### Response:")])

            if answers:
                res = [pred + "---answer---" + a for pred, a in zip(decoded_tokens, answers)]
            else:
                res = decoded_tokens
            results += res

        if i % 10 == 0:
            torch.save(results, "{}_{}.list".format(output_file, rank))
            print("saved to {}_{}.list".format(output_file, rank))

    torch.save(results, "{}_{}.list".format(output_file, rank))
    print("saved to {}_{}.list".format(output_file, rank))


if __name__ == "__main__":
    # DDP init
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    parser.add_argument('--test_file_name_or_path', type=str,
                        default="",
                        help='test file name in the hub or local path')
    parser.add_argument('--saved_file_name_or_path', type=str,
                        default="")
    parser.add_argument('--batch', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--test_size', type=int, default=100000)
    parser.add_argument('--block_size', type=int, default=1024, help='block size')
    parser.add_argument('--temperature', '-t', type=float, default=0.01, help='temperature: 0.7 for text generation')

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--mode", type=str, default=None, help="the type of create_prompt")
    parser.add_argument("--eval_data", type=str, default=None, help="the dataset of evaluation")

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')

    args = parser.parse_args()

    model_suffix = args.model_name_or_path.replace("/checkpoint", "").split("/")[-1]

    args.saved_file_name_or_path = args.saved_file_name_or_path + model_suffix + "_" + args.eval_data.split("/")[-1]

    output_file = args.saved_file_name_or_path
    output_file = output_file

    gpus_per_process = 1
    num_processes = args.gpus // gpus_per_process
    args.world_size = num_processes  #
    mp.spawn(train_gpu, nprocs=num_processes, args=(args, num_processes, gpus_per_process))

    all_result = []

    print("Output path is {}".format(output_file))

    for rank in range(num_processes):
        all_result += torch.load("{}_{}.list".format(output_file, rank))
        os.remove("{}_{}.list".format(output_file, rank))

    torch.save(all_result, "{}.list".format(output_file))
    write_to_json(all_result, "{}.json".format(output_file))
