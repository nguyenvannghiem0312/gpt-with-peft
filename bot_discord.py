import discord
from discord.ext import commands
import os; os.environ["CUDA_DEVICE"] = os.environ.get("CUDA_DEVICE") or "0"

import torch

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cài đặt token Discord và token của mô hình GPT-3
TOKEN = "### Your token"

# Khởi tạo bot với Intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def make_prompt(instruction):
        return f"""Hãy viết một phản hồi thích hợp cho chỉ dẫn dưới đây.

### Instruction:
{instruction}

### Response:"""

BASE_MODEL = "VietAI/gpt-neo-1.3B-vietnamese-news"
PEFT_WEIGHTS = "chat-gpt-neo-1.3B"

print("------Import model------")


model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, torch_dtype=torch.bfloat16)

print("------Import succesful------\n")

if torch.cuda.is_available():
    device = "cuda"
    model.to(device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model.eval()

def get_answer(q, do_sample = True, max_new_tokens=200, top_k=20, temperature=1, skip_tl=False):
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

    
# Sự kiện khi bot đã sẵn sàng
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')

# Sự kiện khi bot nhận tin nhắn
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if message.content.startswith('!ask'):
        user_input = message.content[len('!ask'):].strip()
        # Đảm bảo bot không tự trả lời chính mình

        response = get_answer(user_input)
        # Gửi lại tin nhắn của người dùng
        await message.channel.send(response)

# Chạy bot trên Discord
bot.run(TOKEN)
