import os 
from dotenv import load_dotenv
import config
from game_master import GameMaster
from api_clients import YandexGPTPlayer


def main():
    load_dotenv()

    print("ИНИЦИАЛИЗАЦИ ИГРЫ")
    print("Подключение к нейросетям и раздача ролей...")

    players = [

            YandexGPTPlayer(
                name="Зарина",
                role="mafia", 
                system_prompt=config.get_system_prompt("mafia", "Зарина")
            ),

            YandexGPTPlayer(
                name="Андрей",
                role="villager",
                system_prompt=config.get_system_prompt("villager", "Андрей")
            ),

            YandexGPTPlayer(
                name="Рома",
                role="villager",
                system_prompt=config.get_system_prompt("villager", "Рома")
            ),

            YandexGPTPlayer(
                name="Арслан",
                role="mafia", 
                system_prompt=config.get_system_prompt("mafia", "Арслан")
            ),

            YandexGPTPlayer(
                name="Махмуд",
                role="villager",
                system_prompt=config.get_system_prompt("villager", "Махмуд")
            ),

            YandexGPTPlayer(
                name="Миша",
                role="villager", 
                system_prompt=config.get_system_prompt("mafia", "Миша")
            ),

            YandexGPTPlayer(
                name="Костя",
                role="villager",
                system_prompt=config.get_system_prompt("villager", "Костя")
            ),

            YandexGPTPlayer(
                name="Дима",
                role="villager",
                system_prompt=config.get_system_prompt("villager", "Дима")
            ),
        ]

    game = GameMaster(players)

    game.run_game_loop()

if __name__ == "__main__":
    main()