import os
import requests
import itertools
import random
import re
import threading

class GeminiAgent:
    """
    AI agent for Tiến Lên controlled by Gemini-2.0-Flash LLM.
    Rotates API keys after each call.
    """
    _api_keys = []
    _key_index = itertools.cycle([0])  # fallback; replaced after loading keys

    @classmethod
    def load_api_keys(cls):
        if not cls._api_keys:
            api_keys_str = os.environ.get("GEMENI_API_LIST_KEY", "")
            keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
            cls._api_keys = keys
            cls._key_index = itertools.cycle(range(len(keys))) if keys else itertools.cycle([0])

    @classmethod
    def get_next_api_key(cls):
        cls.load_api_keys()
        if cls._api_keys:
            idx = next(cls._key_index)
            return cls._api_keys[idx]
        return None

    def __init__(self):
        self.model = "gemini-2.0-flash"

    def format_prompt(self, player_hand, play_history, player=None, current_table=None, all_players=None):
        """
        Prepare structured prompt for Gemini, with richer game context.
        Args:
            player_hand (list of Card): Player's hand.
            play_history (list of dict): [{'player': str, 'move': list of Card or "PASS"}]
            player (Player): (Optional) current player object for richer prompts.
            current_table (list of str): Current played cards.
            all_players (list): All player objects (optional, to show player hands left)
        Returns:
            str
        """
        hand_str = ", ".join([f"{c.rank}{c.suit[0].upper()}" for c in player_hand])
        hand_count = len(player_hand)
        # Table state
        if current_table:
            table_str = ", ".join(current_table)
            table_state = f"Bài trên bàn: {table_str} (tổng {len(current_table)} lá)"
        else:
            table_state = "BÀN ĐANG TRỐNG - BẠN PHẢI ĐÁNH BÀI, KHÔNG ĐƯỢC PASS!"

        # Show other players' hand counts if possible
        players_left = ""
        if all_players:
            players_left = "\n- Số bài còn lại của từng người: " + ", ".join(
                [f"{pl.name}: {len(pl.hand)}" for pl in all_players]
            )

        # Lịch sử các nước đi (10 gần nhất)
        moves_str = ""
        for hist in play_history[-10:]:
            player_name = hist.get('player', 'Người chơi')
            move = hist.get('move', [])
            if move == "PASS":
                moves_str += f"{player_name}: PASS\n"
            else:
                move_str = ", ".join(move) if isinstance(move, list) else str(move)
                moves_str += f"{player_name}: {move_str}\n"

        identity = ""
        if player:
            identity = f"Bạn là {player.name} (ID: {player.id}). "

        # LUAT CHOI DAY DU HON
        rules = (
            "\n\nLUẬT CHƠI CƠ BẢN:"
            "\n- Mỗi lượt bạn phải đánh bài lớn hơn bài trên bàn cùng loại (đơn, đôi, sảnh * 3 đến 12 lá, 3 đôi thông * dùng để đánh 1 heo, 4 đôi thông *dùng để đánh 1 heo, đôi heo, 3 đôi thông và tứ quý, tứ quý * đánh được heo, đôi heo, 3 đôi thông) hoặc PASS đẻ đợi thời cơ tiếp theo"
            "\n- Trường hợp nếu không thể đánh hợp lệ thì bắt buộc phải PASS."
            "\n- Không được đánh bài '2' hoặc combo chứa 2 ở lượt đầu tiên trừ khi chỉ còn bài 2."
            "\n- Không được đánh toàn bộ 13 lá bài cùng lúc."
            "\n- Các sảnh phải liên tiếp, không chứa bài 2."
            "\n- Đôi thông (pairs straight): chuỗi các đôi liên tiếp từ 3 đôi trở lên, không chứa bài 2."
            "\n- Ưu tiên đánh bài nhỏ trước để giảm rủi ro bị chặn."
        )
        # ĐẶC BIỆT: Lượt đầu tiên, nếu có 3♠ PHẢI đánh lá này!
        game_first_turn = False
        has_3spades = False
        # Check via play_history for initial round, and hand
        if (play_history is not None and isinstance(play_history, list)
            and len(play_history) == 0 and player_hand is not None):
            game_first_turn = True
        if player_hand is not None:
            has_3spades = any(getattr(card, "rank", None) == "3" and getattr(card, "suit", None) == "spades" for card in player_hand)
        if game_first_turn and has_3spades:
            rules += "\n- ĐẶC BIỆT: Bạn có 3♠ nên BẮT BUỘC phải đánh lá này trong lượt đầu tiên! KHÔNG ĐƯỢC PASS."

        # Added explicit warning about invalid 13-card moves, moved below LUAT
        warning = (
            "\nCẢNH BÁO QUAN TRỌNG:"
            "\n1. Khi bàn trống (không có bài): PHẢI đánh bài hợp lệ, không được PASS."
            "\n2. Không được đánh toàn bộ 13 lá hoặc trả lời sai định dạng, hệ thống sẽ coi là PASS."
            "\n3. Khi bàn đã có bài: chỉ được đánh combo đúng loại và lớn hơn hoặc PASS nếu không có bài phù hợp."
            "\n4. Nếu đánh nhiều hơn 4 lá mà không phải sảnh/đôi thông, coi như PASS."
        )

        move_instructions = (
            "\n\nCÁCH TRẢ LỜI:\n"
            "- Để đánh bài: Liệt kê các lá bài, phân cách bởi dấu phẩy, theo định dạng <rank><suit_initial> (VD: 3S, 5D, 10H)"
            "\n- Để PASS: Ghi chính xác 'PASS' (chữ in hoa)."
            "\nVí dụ hợp lệ:\n    3S\n    4C, 4D\n    5S, 6C, 7D\n    PASS"
            "\nVí dụ KHÔNG hợp lệ:\n    Đôi 3\n    Ba con 5\n    3 bích\n    Pass (thường)"
        )

        suit_mapping = (
            "\nKÝ HIỆU CHẤT BÀI:\n"
            "- S = ♠ (Bích)\n"
            "- C = ♣ (Chuồn)\n"
            "- D = ♦ (Rô)\n"
            "- H = ♥ (Cơ)\n"
        )

        game_state = (
            f"\n\nTRẠNG THÁI HIỆN TẠI:\n- Thứ tự lượt chơi: {', '.join([p.name for p in all_players]) if all_players else ''}"
            f"\n- Lượt số: {len(play_history) + 1}"
            f"\n- {table_state}"
            f"\n- Số bài còn lại trên tay bạn: {hand_count}"
            f"{players_left}"
        )

        return (
            f"{identity}Bạn đang chơi tiến lên miền Nam."
            f"\n{game_state}"
            f"\n\nBÀI TRÊN TAY: {hand_str}"
            f"\n\nLịch sử lượt đi gần nhất:\n" +
            (moves_str if moves_str else "(Chưa có nước đi nào)\n") +
            rules +
            warning +
            suit_mapping +
            move_instructions +
            "\n\nTRẢ LỜI (chỉ ghi bài hoặc PASS):"
        )

    def parse_llm_response(self, llm_response, player_hand):
        """
        Parse Gemini's response with improved error handling and card extraction.
        """
        from .player import Player
        text = llm_response.strip().upper()

        # Handle PASS (must be exactly "PASS", not "pass" etc)
        if "PASS" in text:
            return "PASS"

        valid_cards = []
        # Use regex split for better separation
        tokens = re.split(r'[\s,;.]+', text)
        for token in tokens:
            token = token.strip()
            if not token:
                continue

            # Special case: handle "10" rank
            if token.startswith("10") and len(token) > 2:
                rank = "10"
                suit_code = token[2]
            elif len(token) >= 2:
                rank = token[:-1]
                suit_code = token[-1]
            else:
                continue

            suit_map = {"S": "spades", "C": "clubs", "D": "diamonds", "H": "hearts"}
            suit = suit_map.get(suit_code)
            # Must match rank and suit, rank is always string, and check it's in Card.RANKS
            if not suit or rank not in [c.rank for c in player_hand]:
                continue
            # Find card in hand and add if not already used
            for card in player_hand:
                if card.rank == rank and card.suit == suit and card not in valid_cards:
                    valid_cards.append(card)
                    break

        # Validate using Player.get_combo_type; force PASS if invalid or 13 cards (shouldn't, but safety)
        if valid_cards:
            combo_type = Player.get_combo_type(valid_cards)
            if combo_type == "INVALID" or len(valid_cards) == 13:
                return "PASS"
        return valid_cards if valid_cards else "PASS"

    def select_action(self, game, player):
        """
        Given game and player, call Gemini to pick next move. Returns "PASS" or list of Card.
        Adds detailed log for debugging.
        """
        from src.game.card import Card

        # Add move_history if not exists
        if not hasattr(game, 'move_history'):
            game.move_history = []

        player_hand = player.hand[:]
        play_history = game.move_history[:]
        current_table = [f"{c.rank}{c.suit[0].upper()}" for c in getattr(game, "last_played_cards", [])] if getattr(game, "last_played_cards", None) else []
        # Provide all players so LLM may reason about counts/order (optional but helpful)
        all_players = getattr(game, "players", None)

        prompt = self.format_prompt(
            player_hand,
            play_history,
            player=player,
            current_table=current_table,
            all_players=all_players
        )

        api_key = GeminiAgent.get_next_api_key()
        key_str = api_key[:6] + "..." if api_key else "NONE"
        if not api_key:
            return "PASS"

        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + api_key

        from src.game.main import debug_log 

        retry_count = 0
        max_retries = 2
        while retry_count < max_retries:
            try:
                data = {}
                llm_answer = ""
                # --- RUN API REQUEST IN SEPARATE THREAD TO AVOID BLOCK GUI ---
                def api_call():
                    nonlocal data, llm_answer
                    try:
                        resp = requests.post(api_url, json=payload, timeout=15, headers=headers)
                        data = resp.json()
                        if "candidates" in data and data["candidates"]:
                            llm_answer = data["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            llm_answer = data.get("text", "")
                    except Exception as e:
                        pass

                api_thread = threading.Thread(target=api_call)
                api_thread.start()
                api_thread.join(timeout=15)
                # Bắt buộc stop nếu chưa có dữ liệu sau timeout tránh treo
                if api_thread.is_alive():
                    llm_answer = ""

            except Exception as e:
                return "PASS"

            result = self.parse_llm_response(llm_answer, player_hand)

            # --- Fix: Block PASS if table is empty ---
            if not getattr(game, "last_played_cards", None) and result == "PASS":
                retry_count += 1
                continue  # Force retry
            # ------------------------------------------------------
            return result

        # If failed too many times, play lowest card as fallback
        if not getattr(game, "last_played_cards", None) and player_hand:
            return [min(player_hand, key=lambda c: (Card.RANKS.index(c.rank), c.suit))]
        return "PASS"