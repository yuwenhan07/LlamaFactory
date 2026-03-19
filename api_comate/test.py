from openai import OpenAI
import os
client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key=os.environ['QIANFAN_API_KEY'],
)
response = client.chat.completions.create(
    model="kimi-k2.5", 
    messages=[{"role":"user","content":"你好"}],
    extra_body={ 
        "stop":[], 
        "enable_thinking":True
    }
)
print(response)