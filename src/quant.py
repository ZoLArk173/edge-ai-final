from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from gptqmodel import GPTQModel, QuantizeConfig, BACKEND

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

model_id = "meta-llama/Llama-3.2-3B-Instruct"
quant_path = "llama-3.2-3b-instruct-wcowen-quant"

tokenizer = AutoTokenizer.from_pretrained(model_id)
calibration_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4,group_size=32, sym=False,)

model = GPTQModel.load(model_id, quant_config, torch_dtype=torch.float16)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)
tokenizer.save_pretrained(quant_path)

print("Loading quantized model...")
model = GPTQModel.load(quant_path)

# Evaluate PPL
ppl = evaluate_ppl(model, tokenizer)
print(f"Perplexity: {ppl}")
