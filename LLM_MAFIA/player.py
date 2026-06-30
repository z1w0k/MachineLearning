import json

class Player:
    def __init__(self, name, model_name, role, system_prompt):
        self.name = name
        self.model_name = model_name
        self.role = role
        self.is_alive = True

        self.messages = [
            {"role": "system", 
             "content": system_prompt}
        ]

    def receive_message(self, text):
        self.messages.append({"role": "user", 
                              "content": text})

    def generate_response(self):
        pass

    def parse_json(self, raw_text):
        try:
            start = raw_text.find('{')
            end = raw_text.find('}') + 1
            clean_json = raw_text[start:end]
            return json.loads(clean_json)
        except (ValueError, json.JSONDecodeError):
            print(f"Ошибка парсинга JSON у игрока {self.name}. Сырой текст {raw_text}")
            return {"thoughts": "Произошла ошибка в моей голове.", "chat": "Я в замешательстве.", "vote": "None"}
        
