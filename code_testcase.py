import transformers
from transformers import AutoTokenizer, GemmaTokenizer, AutoModelForCausalLM, set_seed
import torch
import json
import re
import os
import random
from tqdm import tqdm
from openai import OpenAI
from utils import get_model, get_prompt_testcase, API_KEY, REPEAT_T, TEST_DATA

root_path = os.path.dirname(__file__)
DEVICE = 'cuda:1'

def reorder_string(s, seed):
    lines = s.strip().split('\n')
    # Keep the first two lines unchanged
    first_two_lines = lines[:2]
    # Randomly shuffle the rest of the lines
    remaining_lines = lines[2:]
    random.seed(seed)
    random.shuffle(remaining_lines)
    result = '\n'.join(first_two_lines + remaining_lines)
    
    return result

def write_data(datasets, file_name):
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        json.dump(datasets, file)
        file.write('\n')


if __name__ == "__main__":
    data = TEST_DATA

        
    for k,d in enumerate(data):
        
        topic = d['topic']
        category = d['cat']
        for code_type in ['code', 'code_form', 'code_pl', 'code_order']:
            code = d[code_type]

            folder_name = f'{root_path}/testcase/{category}/'
            write_file_name = folder_name + f'{topic}.jsonl'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)       

            for j in range(REPEAT_T):
                # get question with random order
                question = reorder_string(d['question'], seed=j)
                full_question = question + code
               
                prompt = get_prompt_testcase(full_question)
               
               
                messages=[
                    {"role": "system", "content": "You are a helpful agent.\n{{extra_info}}"},
                    {
                        "role": "user",
                        "content": prompt
                    }
                    ]            
                
                write_data(messages, write_file_name)
