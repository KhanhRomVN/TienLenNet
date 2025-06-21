import random

class VirtualChef:
    """
    Đề xuất nước đi hợp lệ ngẫu nhiên cho AI để tăng exploration
    """
    @staticmethod
    def suggest_random_valid_move(valid_actions):
        """
        Trả về một action_id ngẫu nhiên từ danh sách hợp lệ (ưu tiên không chọn PASS nếu có thể)
        :param valid_actions: list action_id, mặc định valid_actions[0] là PASS
        """
        if not valid_actions:
            return 0
        if len(valid_actions) == 1:
            return valid_actions[0]
        if valid_actions[0] == 0:
            return random.choice(valid_actions[1:]) if len(valid_actions) > 1 else 0
        return random.choice(valid_actions)