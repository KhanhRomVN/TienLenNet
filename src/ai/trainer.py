import numpy as np
from .agent import AIAgent
from src.game.main import TienLenGame
from .human_dataset import HumanDemonstrationDataset, pretrain_with_human_data

class Trainer:
    def __init__(self):
        # Chạy với headless rendering để train hiệu suất cao (không tạo UI pygame)
        self.game = TienLenGame(render=False)
        self.agent = AIAgent(0)
        self.episodes = 10000
        self.batch_size = 64
        self.update_target_every = 100
def pretrain_from_human_logs(self, game_records, epochs=50):
        """
        Huấn luyện sơ bộ agent từ dữ liệu người chơi thật (Giai đoạn 1)
        game_records: list các log dạng {'state': ..., 'action_id': ...}
        """
        dataset = HumanDemonstrationDataset(game_records, encode_state_func=lambda s: self.agent.state_encoder.encode_from_log(s))
        pretrain_with_human_data(self.agent, dataset, self.agent.device, epochs=epochs)

    def train(self):
        for episode in range(self.episodes):
            self.game.prepare_new_game()
            state = self.agent.state_encoder.encode(self.game, 0)
            total_reward = 0
            done = False

            while not done:
                # Chọn hành động
                action_id = self.agent.agent.act(state)
                action = self.agent.interpret_action(action_id, self.game)

                # Thực hiện hành động
                prev_state = state
                reward, done = self.execute_action(action)
                next_state = self.agent.state_encoder.encode(self.game, 0)
                total_reward += reward

                # Lưu dữ liệu vào bộ nhớ replay
                # LOG replay buffer push cho mọi action (đặc biệt action_id != 0)
                if action_id != 0:
                    print(f"[TRAINER LOG] PUSH REPLAY: action_id={action_id}, reward={reward}")
                else:
                    print(f"[TRAINER LOG] PUSH REPLAY: PASS action")
                self.agent.agent.memory.push(
                    prev_state,
                    action_id,
                    reward,
                    next_state,
                    done
                )

                # Huấn luyện model
                self.agent.agent.update_model(self.batch_size)
                state = next_state

            # Cập nhật target net định kỳ
            if episode % self.update_target_every == 0:
                self.agent.agent.update_target_net()

            # Giảm epsilon
            self.agent.agent.update_epsilon()

            print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {self.agent.agent.epsilon:.4f}")

        # Lưu mô hình đã train
        self.agent.agent.save("models/tien_len_ai.pth")

    def execute_action(self, action):
        """Thực hiện action và tính reward (bao gồm chiến thuật)"""
        player = self.game.players[0]
        prev_score = player.score

        total_moves = getattr(self, "total_moves", 0)
        if not hasattr(self, "explored_combos"):
            self.explored_combos = set()

        if action == "PASS":
            self.game.pass_turn()
        else:
            # Chọn lá bài tương ứng
            for card in player.hand:
                card.selected = card in (action if isinstance(action, list) else [])
            self.game.play_cards()
            selected_count = len([c for c in player.hand if c.selected])
            # Không dùng reward cứng, thay bằng get_dynamic_reward cho chiến lược thông minh hơn
            if selected_count > 0:
                base_reward = player.get_dynamic_reward(action, self.game, total_moves=total_moves, explored_combos=self.explored_combos)
                player.add_reward(base_reward, "Dynamic combo reward")
                if selected_count == 1:
                    player.add_reward(0.3, "Thoát bài khó")
            # Chiến thuật layer
            player.update_strategy_rewards()

        reward = player.score - prev_score
        self.total_moves = total_moves + 1
        done = self.game.round_starter is None  # Game kết thúc hoặc sang ván mới

        return reward, done