# 撰写一个调用openai的Python脚本，测试demon

import openai
api_key = "sk-e346078d76f546c2ab04f0f008126a91"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "deepseek-r1"
def test_demon_prompt(prompt):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )
    return response.choices[0].message.content
    
if __name__ == "__main__":
    demon_prompt = "Explain the concept of demon prompts in AI language models."
    result = test_demon_prompt(demon_prompt)
    print("Response from GPT-4:")
    print(result)
    