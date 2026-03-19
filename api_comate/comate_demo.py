from openai import OpenAI
import os
client = OpenAI(
    base_url='https://oneapi-comate.baidu-int.com/v1',
    api_key=os.getenv('COMATE_API_KEY')
)
response = client.chat.completions.create(
    model="Claude Sonnet 4.6", 
    messages=[{"role":"user","content":"你好"}], 
    temperature=0, 
    extra_body={ 
        "stop":[], 
        "enable_thinking":True
    }
)
print(response)