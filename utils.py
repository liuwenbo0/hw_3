import json
import collections
import string
import re
from rouge_score import rouge_scorer
import requests
from metrics import qa_f1_zh_score_all

def scorer_all(dataset, predictions, answers):
    score = 0.
    f1, pr, recall = qa_f1_zh_score_all(predictions, answers)
    return f1, pr, recall

def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path,encoding='utf-8') as f:
        return json.load(f)

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def parse_generation(s):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def build_qa_prompt(example, query_prompt):
    q = example["question"]
    doc_prompts = [f"<｜User｜>{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    q_prompt = f"{query_prompt}{q}\n回答：<｜Assistant｜><think>\n"
    return doc_prompts, q_prompt

def extract_after_think(text):
    marker = '</think>\n\n'
    index = text.find(marker)
    if index != -1:
        return text[index + len(marker):]
    else:
        return ''