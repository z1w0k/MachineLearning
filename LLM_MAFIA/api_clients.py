import os
import json
from openai import OpenAI
from player import Player
import requests

'''class DeepSeekPlayer(Player):
    def __init__(self, name, role, system_prompt):
        super().__init__(name = name, model_name= 'deepseek', role= role, system_prompt= system_prompt)


        self.client = OpenAI(
            api_key= os.getenv("DEEPSEEK_API_KEY"),
            base_url= "https://api.deepseek.com"
        )

    def generate_response(self):
        try:
            response = self.client.chat.completions.create(
                model = "deepseek-chat",
                messages= self.messages,
                response_format= {"type": "json_object"},
                temperature= 0.7
            )

            raw_text = response.choices[0].message.content
 
            self.messages.append({"role": "assistant",
                                  "content": raw_text})
            
            return self.parse_json(raw_text)
        
        except Exception as e:
            print(f"[{self.name} / DeepSeek] Ошибка API: {e}")
            return {"thoughts": "Связь прервалась.",
                    "chat": "...",
                    "vote": "null"}'''

'''class OpenRouterPlayer(Player):
    def __init__(self, name, role, system_prompt, model_id="google/gemma-2-9b-it:free"):
        super().__init__(name=name, model_name=model_id, role=role, system_prompt=system_prompt)
        
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_id = model_id

    def generate_response(self):
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=self.messages,
                response_format={"type": "json_object"},
                temperature=0.7 
            )
            raw_text = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": raw_text})
            return self.parse_json(raw_text)
            
        except Exception as e:
            print(f"[{self.name} / OpenRouter] Ошибка API: {e}")
            return {"thoughts": "Связь прервалась.", "chat": "...", "vote": "null"}'''

'''class GigaChatPlayer(Player):
    def __init__(self, name, role, system_prompt):
        super().__init__(name=name, model_name='gigachat', role=role, system_prompt=system_prompt)
        self.credentials = os.getenv("GIGACHAT_CREDENTIALS")

    def generate_response(self):
        try:
            with GigaChat(credentials=self.credentials, verify_ssl_certs=False) as giga:
                response = giga.chat({
                    "messages": self.messages,
                    "temperature": 0.7
                })
                
                raw_text = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": raw_text})
                return self.parse_json(raw_text)
                
        except Exception as e:
            print(f"[{self.name} / GigaChat] Ошибка API: {e}")
            return {"thoughts": "Я временно потерял сознание.", "chat": "...", "vote": "null"}'''
        
class YandexGPTPlayer(Player):
    def __init__(self, name, role, system_prompt):
        super().__init__(name=name, model_name='yandexgpt', role=role, system_prompt=system_prompt)
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")

    def generate_response(self):
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        yandex_messages = []
        for msg in self.messages:
            yandex_messages.append({
                "role": msg["role"],
                "text": msg["content"]
            })

        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": 0.7,
                "maxTokens": "2000"
            },
            "messages": yandex_messages
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response_data = response.json()
            
            if "result" not in response_data:
                print(f"\n[{self.name} / YandexGPT] ВНИМАНИЕ, ОТВЕТ СЕРВЕРА:")
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
                return None
            
            raw_text = response_data["result"]["alternatives"][0]["message"]["text"]
            self.messages.append({"role": "assistant", "content": raw_text})
            
            return self.parse_json(raw_text)
            
        except Exception as e:
            print(f"[{self.name} / YandexGPT] Ошибка кода/сети: {e}")
            return None