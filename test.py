from vllm import LLM, SamplingParams
from utils import load_dataset, build_qa_prompt, scorer_all, extract_after_think
from transformers import AutoTokenizer
import torch
import torch._dynamo
import os
import sys
import numpy as np
import json

# Disable PyTorch compilation to avoid CUDA linker issues
torch._dynamo.config.suppress_errors = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"

base_dir = ""  ###保存结果路径设置
num_runs = 1
tp = 2  # Use both GPUs for tensor parallel inference
max_tokens = 2048
dataset_name = "just_for_test"
model = "DeepSeek-R1-Distill-Qwen-14B"
model_path = f"data/{model}"

json_path = os.path.join(base_dir, f"{model}-{dataset_name}-test.json")
eval_dataset = load_dataset(f"data/{dataset_name}.json")
if dataset_name == "just_for_test":
    prefix_prompt = "你是一个有帮助且知识渊博的助手。你将得到一个问题和一组从知识库中检索到的文档。请仅使用提供的上下文信息来回答问题。如果上下文中没有足够的信息来回答问题，请如实地说明。需遵从下面的指令：1、你将得到一个用户问题和一组检索到的文档。2、仅使用提供的上下文来回答问题。3、如果问题无法在上下文中找到答案，请回答：“上下文没有提供足够的信息来回答这个问题。”4、简洁并事实性回答。\n文章：\n"
    query_prompt = "请基于上述文章回答下面的问题。\n问题："
llm = LLM(
    model=model_path,
    max_model_len=20000,
    tensor_parallel_size=tp,
    enforce_eager=True,
    enable_chunked_prefill=False,
    dtype="bfloat16",
    disable_custom_all_reduce=True,  # Fix multi-GPU communication issues
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_config = load_dataset(f"{model_path}/config.json")
num_layer = model_config["num_hidden_layers"]

ttft_blend = []
answers = []
result_w_caches = []
t_df1 = []
t_dpr = []
t_drecall = []
with open(json_path, mode="a", newline="", encoding="utf-8") as file:
    file.write("[\n")
    for ii, ex in enumerate(eval_dataset):
        dict_obj = {}
        # if ii <= num_runs -2:
        #    continue
        if ii == num_runs:
            break
        answer = ex["answers"]
        question = ex["question"]

        doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
        doc_list = [prefix_prompt] + doc_prompts + [q_prompt]

        sampling_params = SamplingParams(temperature=0, max_tokens=1)
        recompute_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.recompute_metadata
        ####设置各个文档token数，包括开头结尾的prompt
        doc_length = [len(tokenizer.encode(prefix_prompt))]
        for doc in doc_prompts:
            doc_length.append(len(tokenizer.encode(doc)) - 1)
        doc_length.append(len(tokenizer.encode(q_prompt)) - 1)
        recompute_metadata["doc_length"] = doc_length
        recompute_metadata["kv_done"] = False
        chunk_past_key_values = []
        ####提前计算单文档的KV作为缓存
        for i in range(len(doc_list)):
            prompts = doc_list[i]
            llm.generate(prompts, sampling_params)
            llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers

            for j in range(num_layer):
                past_key_values = llm_layers[j].self_attn.hack_kv
                if i == 0:
                    temp_k = past_key_values[0][:].clone()
                    temp_v = past_key_values[1][:].clone()
                    chunk_past_key_values.append([temp_k, temp_v])
                else:
                    temp_k = past_key_values[0][1:].clone()
                    temp_v = past_key_values[1][1:].clone()
                    chunk_past_key_values[j][0] = torch.cat(
                        (chunk_past_key_values[j][0], temp_k), dim=0
                    )
                    chunk_past_key_values[j][1] = torch.cat(
                        (chunk_past_key_values[j][1], temp_v), dim=0
                    )
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = (
            chunk_past_key_values
        )

        ###标记缓存准备就绪
        recompute_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.recompute_metadata
        recompute_metadata["kv_done"] = True

        ###开始推理
        prompts = ""
        for doc in doc_list:
            prompts += doc

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=False
        )
        output = llm.generate(prompts, sampling_params)
        recompute_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.recompute_metadata

        ###计算重算率，超过0.3只返回结果，不计入成绩
        valid_list = recompute_metadata["valid_list"]
        recompute_num = 0
        for u in valid_list:
            recompute_num += u
        total_num = np.sum(doc_length[1:-1]) * num_layer
        recompute_ratio = recompute_num / total_num
        if recompute_ratio >= 0.3:
            print(f"Error!recompute ratio {recompute_ratio} is too large!")

        ###计算结果精度，仅返回每个题的精度给选手，其余不返回
        res = output[0].outputs[0].text
        print(f"问题: {question}")
        print(f"模型答案: {res}")
        print(f"正确答案: {answer}")
        ttft = (
            output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
        )
        print(f"TTFT with cache: {ttft}")
        ttft_blend.append(ttft)
        result_w_caches.append(res)
        answers.append(answer)

        temp_df1 = 0
        temp_dpr = 0
        temp_drecall = 0
        for j in range(len(answer)):
            df1, dpr, drecall = scorer_all(
                "dureader_all", extract_after_think(res), str(answer[j])
            )
            if df1 > temp_df1:
                temp_df1 = df1
            if dpr > temp_dpr:
                temp_dpr = dpr
            if drecall > temp_drecall:
                temp_drecall = drecall
        df1 = temp_df1
        dpr = temp_dpr
        drecall = temp_drecall
        t_df1.append(df1)
        t_dpr.append(dpr)
        t_drecall.append(drecall)

        dict_obj["id"] = ii
        dict_obj["Query"] = question
        dict_obj["Model Answer"] = res
        dict_obj["Dateset Answer"] = answer
        dict_obj["F1 with Dataset"] = df1
        dict_obj["Precision with Dataset"] = dpr
        dict_obj["Recall with Dataset"] = drecall
        dict_obj["TTFT"] = ttft

        json.dump(dict_obj, file, indent=4, ensure_ascii=False)
        file.write(",\n")
    res_obj = {}
    res_obj["avg_ttft"] = np.mean(ttft_blend)
    res_obj["avg_f1 with Dataset Answer"] = np.mean(t_df1)
    res_obj["avg_precision with Dataset Answer"] = np.mean(t_dpr)
    res_obj["avg_recall with Dataset Answer"] = np.mean(t_drecall)
    json.dump(res_obj, file, indent=4, ensure_ascii=False)
    file.write("]\n")
print(
    f"f1: {np.mean(t_df1)}, precision: {np.mean(t_dpr)}, recall: {np.mean(t_drecall)}, ttft: {np.mean(ttft_blend)}"
)
