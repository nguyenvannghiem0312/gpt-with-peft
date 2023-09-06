import os; os.environ["CUDA_DEVICE"] = os.environ.get("CUDA_DEVICE") or "0"

import torch

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def make_prompt(instruction):
        return f"""Hãy viết một phản hồi thích hợp cho chỉ dẫn dưới đây.

### Instruction:
{instruction}

### Response:"""
# END generate_qna_prompt

BASE_MODEL = "VietAI/gpt-neo-1.3B-vietnamese-news"
PEFT_WEIGHTS = "chat-gpt-neo-1.3B"
load_in_8bit = False

print("------Import model------")


model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, torch_dtype=torch.bfloat16)

print("------Import succesful------\n")

if torch.cuda.is_available():
    device = "cuda"
    model.to(device)
#   device_map = {'': 0}
#   if load_in_8bit:
#       model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, torch_dtype=torch.bfloat16, device_map=device_map)
#       model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, torch_dtype=torch.bfloat16, device_map=device_map)
#   else:
#       # model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device_map)
#       # model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, torch_dtype=torch.bfloat16, device_map=device_map)
      
# else:
#   device = "cpu"
#   model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
#   model = PeftModel.from_pretrained(model, PEFT_WEIGHTS)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model.eval()

print("-----Configuration parameters for inference------\n")
print("---Do sample (0 or 1) = ", end = '')
do_sample = True if int(input()) > 0 else False
print("---Max new tokens (integer >= 0) = ", end = '')
max_new_tokens = int(input())
print("---Top K (integer >= 0) = ", end = '')
top_k = int(input())
print("---Temperature (float >= 0) = ", end = '')
temperature = float(input())

print("\n\n\n-----Hello World------")

def get_answer(q, do_sample = True, max_new_tokens=196, top_k=20, temperature=1, skip_tl=False):
  input_ids = tokenizer(make_prompt(q), return_tensors="pt")["input_ids"].to(device)
  with torch.no_grad():
      gen_tokens = model.generate(
          input_ids=input_ids,
          max_length=len(input_ids) + max_new_tokens,
          do_sample=do_sample,
          temperature=temperature,
          top_k=top_k,
          repetition_penalty=1.2,
          eos_token_id=0, # for open-end generation.
          pad_token_id=tokenizer.eos_token_id,
      )
  origin_output = tokenizer.batch_decode(gen_tokens)[0]
  output = origin_output.split("###")[2]
  try:
      k = output.index(":")
      if k < 10: output = output[k+1:]
  except:
      output = output
  # print(f"\n- - -{origin_output}- - -\n")
  return output.strip()[:-13]

while True:
  query = input("\n@Bạn: ")
  print(f"@ChatBot: {get_answer(query, do_sample=do_sample, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)}")



