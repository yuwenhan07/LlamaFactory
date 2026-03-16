# 请安装 OpenAI SDK : pip install openai
# apiKey 获取地址： https://console.bce.baidu.com/qianfan/ais/console/apiKey
# 支持的模型列表： https://cloud.baidu.com/doc/qianfan-docs/s/7m95lyy43

from openai import OpenAI
import os
client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key=os.getenv('QIANFAN_API_KEY')
)
response = client.chat.completions.create(
    model="qwen3.5-397b-a17b", 
    messages=[{"role":"user","content":"你好"}], 
    temperature=0.6, 
    top_p=0.95,
    extra_body={ 
        "stop":[], 
        "enable_thinking":True
    }
)
print(response)