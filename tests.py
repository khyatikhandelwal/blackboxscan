from blackboxscan.scanner import GenerativeModelOutputs
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
scanner = GenerativeModelOutputs(
    model=model, tokenizer=tokenizer, inputs="This is the context."
)
res = scanner.sentence_log_likelihoods(words="Brahmin")
print(res.get_tokens())
print(res.get_total())
