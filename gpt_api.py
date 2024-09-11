import requests
import asyncio
import aiohttp
import os

OPENAI_API_BASE = "https://api.openai.com/v1/chat/completions"

class PromptBuilder:
    @staticmethod
    def equip_QA(template_path: str, model_response: str, user_prompt: str=None):
        with open(template_path) as f:
            prompt_template = f.read()
        
        prompt_template = prompt_template.format(model_response=model_response)

        if user_prompt is not None:
            prompt_template = prompt_template.format(user_prompt=user_prompt)
        
        return prompt_template

    @staticmethod
    def equip_INST(template_path: str, instruction: str):
        with open(template_path) as f:
            prompt_template = f.read()
        return prompt_template.format(instruction=instruction)

class GPT:
    @staticmethod
    async def query(
        api_name: str, 
        user_prompt: str, 
        temperature: float=0.7, 
        max_tokens: int=100
    ):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENAI_API_BASE,
                headers=headers,
                json={
                    "model": api_name,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to query GPT: {response}")
                return (await response.json())['choices'][0]['message']['content']
    
    @staticmethod
    async def batch_query(
        api_name: str, 
        user_prompts: list, 
        temperature: float=0.7, 
        max_tokens: int=100
    ):
        tasks = [
            GPT.query(
                api_name=api_name,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for user_prompt in user_prompts
        ]
        return await asyncio.gather(*tasks)
    