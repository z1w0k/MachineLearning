import time
from collections import Counter

class GameMaster:
    def __init__(self, players):
        self.players = players
        self.day_count = 1
        self.game_over = False

    def get_alive_players(self):
        return [p for p in self.players if p.is_alive]
    
    def broadcast_message(self, message, exclude_player=None):
        for player in self.get_alive_players():
            if player != exclude_player:
                player.receive_message(message)

    def kill_player(self, player_name, reason):
        for player in self.get_alive_players():
            if player.name == player_name:
                player.is_alive = False
                announcement = f"Игрок {player.name} был {reason}. Его роль была: {player.role}."
                print(f"\n[ВЕДУЩИЙ]: {announcement}")
                self.broadcast_message(announcement)
                return True
        return False

    def check_win_condition(self):
        alive_players = self.get_alive_players()
        mafia_count = sum(1 for p in alive_players if p.role == 'mafia')
        village_count = len(alive_players) - mafia_count

        if mafia_count == 0:
            print("\nИГРА ОКОНЧЕНА: Мирные жители победили!")
            self.game_over = True
        elif mafia_count >= village_count:
            print("\nИГРА ОКОНЧЕНА: Мафия победила!")
            self.game_over = True

    def phase_day_discussion(self):
        print(f"\n--- ДЕНЬ {self.day_count}: ОБСУЖДЕНИЕ ---")
        self.broadcast_message(f"Наступил день {self.day_count}. Город просыпается. Вы должны вычислить мафию. У каждого есть одна реплика.")
        
        alive_players = self.get_alive_players()
        if not alive_players:
            return

        shift = (self.day_count - 1) % len(alive_players)
        speaking_order = alive_players[shift:] + alive_players[:shift]

        for player in speaking_order:
            print(f"Ожидаем ответ от {player.name}...")
            response_data = player.generate_response()
            
            if response_data:
                chat = response_data.get("chat", "")
                suspect = response_data.get("suspect", "никого")
                
                print(f"[{player.name}]: {chat}")
                print(f"  -> ПОДОЗРЕВАЕТ: {suspect}")
                
                self.broadcast_message(f"Игрок {player.name} говорит: {chat}. Его подозрение: {suspect}")
            
            input("\n")

    def phase_voting(self):
        print(f"\n--- ДЕНЬ {self.day_count}: ГОЛОСОВАНИЕ ---")
        
        alive_players = self.get_alive_players()
        alive_names = [p.name for p in alive_players]
        
        self.broadcast_message(
            f"Обсуждение окончено. Время голосовать! Живые игроки: {', '.join(alive_names)}. "
            f"Напишите в поле 'vote' имя того, против кого вы голосуете (кто должен покинуть город). "
            f"В поле 'chat' можете кратко объяснить причину."
        )

        votes = []

        for player in alive_players:
            print(f"Ожидаем голос от {player.name}...")
            response_data = player.generate_response()
            
            if response_data is None:
                print(f"[{player.name}]: пропустил голосование (ошибка связи).")
                continue 
                
            chat_reason = response_data.get("chat", "")
            if chat_reason and chat_reason.lower() != "null" and chat_reason != "...":
                print(f"[{player.name} объясняет свой голос]: {chat_reason}")

            vote = response_data.get("vote", "")
            if vote is None: vote = ""
            vote = str(vote).strip().strip(".")
            
            if vote in alive_names:
                votes.append(vote)
                print(f" -> {player.name} голосует против: {vote}")
            else:
                print(f" -> {player.name} не выбрал кандидата или выбрал того, кого нет.")

            input(f"\n")

        if votes:
            vote_counts = Counter(votes)
            print(f"\nИтоги голосования: {dict(vote_counts)}")
            top_votes = vote_counts.most_common(2)
            
            if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]:
                print("Голоса разделились поровну! По правилам города сегодня никто не будет казнен.")
                self.broadcast_message("Голоса горожан разделились поровну. Казнь отменяется.")
            else:
                lynched_player = top_votes[0][0]
                print(f"Большинством голосов горожане решили выгнать: {lynched_player}!")
                self.kill_player(lynched_player, "исключен на дневном голосовании")
        else:
            print("Никто не проголосовал. Город в замешательстве.")
            self.broadcast_message("Никто не проголосовал. Казнь отменяется.")
            
        input("\n")

    def phase_night(self):
        print(f"\n--- НОЧЬ {self.day_count}: МАФИЯ ВЫБИРАЕТ ЖЕРТВУ ---")
        self.broadcast_message("Наступает ночь. Город засыпает. Мафия выходит на охоту.")

        mafia_players = [p for p in self.get_alive_players() if p.role == 'mafia']

        if not mafia_players:
            print("В городе не осталось мафии.")
            return

        print(f"В заговоре участвуют: {', '.join([m.name for m in mafia_players])}")
        
        night_votes = []

        for mafia in mafia_players:
            print(f"Мафия {mafia.name} делает свой выбор...")
            
            partners = [p.name for p in mafia_players if p != mafia]
            partners_str = f"Твои напарники: {', '.join(partners)}" if partners else "Ты остался один."
            
            mafia.receive_message(
                f"Сейчас ночь. {partners_str} Напиши в поле 'vote' имя игрока (из мирных), "
                f"которого ваша команда хочет тайно вывести из игры. В поле 'chat' передай null."
            )
            
            response_data = mafia.generate_response()
            
            if response_data is None:
                print(f"Мафиози {mafia.name} заснул на дежурстве (ошибка API).")
            else:
                target = response_data.get("vote", "")
                if target and target.lower() != "null" and target != "...":
                    night_votes.append(target)
                    print(f"Мафия {mafia.name} проголосовала за исключение {target}")

        if night_votes:
            vote_counts = Counter(night_votes)
            top_votes = vote_counts.most_common(2)
            final_target = None
            
            if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]:
                print(f"Назрел раскол! Голоса мафии разделились: {dict(vote_counts)}")
                final_target = night_votes[1]
                print(f"По правилу синдиката, активирован выбор ВТОРОГО проголосовавшего: {final_target}")
            else:
                final_target = top_votes[0][0]
                print(f"Мафия пришла к согласию. Выбрана цель: {final_target}")
            
            killed = self.kill_player(final_target, "выбыл из игры этой ночью")
            if not killed:
                print(f"Мафия промахнулась! Игрок с именем '{final_target}' не найден или уже мертв.")
        else:
            print("Мафия никого не убила.")

    def run_game_loop(self):
        print("Игра начинается!")
        input("\n")

        self.phase_day_discussion()

        self.phase_night()

        print("\n[ВЕДУЩИЙ]: Ночь подошла к концу. Готовимся к новому дню...")
        input("\n")

        self.day_count += 1

        while not self.game_over:
            self.phase_day_discussion()

            self.phase_voting()
            self.check_win_condition()
            if self.game_over:
                break

            self.phase_night()
            self.check_win_condition()
            if self.game_over:
                break

            print("\n[ВЕДУЩИЙ]: Ночь подошла к концу. Готовимся к новому дню...")
            input("\n")

            self.day_count += 1