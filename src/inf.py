import torch
import random
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    torch.manual_seed(0)
    random.seed(0)

    model_name = "./llama-3.2-3b-instruct-wcowen-quant"
    max_new_tokens = 256
    device = 'cuda:0'

    # Initialize vLLM LLM engine
    llm = LLM(model=model_name, dtype="float16", max_model_len=2048,)
    tokenizer = llm.get_tokenizer()

    warmup_prompt = "Explain what AI is."
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

    # Warm-up phase
    for _ in tqdm(range(5), desc="Warm Up..."):
        _ = llm.generate([warmup_prompt], sampling_params)

    prompt = "How to learn a new language?"
    tputs = []
    time_record = []

    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)

        token_count = len(tokenizer.encode(response))
        tput = token_count / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)

    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')

    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    print(f'Throughput: {org_tput} toks/s')

if __name__ == '__main__':
    main()
