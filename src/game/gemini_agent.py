import os
import requests
import itertools
import random

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

    def format_prompt(self, player_hand, play_history):
        """
        Prepare structured prompt for Gemini.
        Args:
            player_hand (list of Card): Player's hand.
            play_history (list of dict): [{'player': str, 'move': list of Card}]
        Returns:
            str
        """
        hand_str = ", ".join([f"{c.rank}{c.suit[0].upper()}" for c in player_hand])
        moves_str = ""
        for hist in play_history[-10:]:
            play_str = ", ".join([f"{c.rank}{c.suit[0].upper()}" for c in hist.get("move", [])])
            moves_str += f"{hist['player']}: {play_str}\n"
        return (
            "Bạn là AI chơi tiến lên. Ở lượt này bạn có các lá bài: " +
            hand_str +
            "\nLịch sử bàn gần nhất:\n" +
            moves_str +
            "\nHãy chọn các lá bài bạn muốn đánh (ví dụ: 3S, 4S hoặc PASS nếu bỏ lượt). Chỉ trả lời chính xác danh sách bài hoặc PASS."
        )

    def parse_llm_response(self, llm_response, player_hand):
        """
        Parse Gemini's response. Expecting: '3S, 4S', 'PASS', etc.
        Returns actual list of Card (from hand) for move.
        """
        text = llm_response.strip().upper()
        if text.startswith("PASS"):
            return "PASS"
        moves = []
        tokens = [token.strip() for token in text.replace(",", " ").replace("  ", " ").split()]
        for token in tokens:
            if len(token) < 2: continue
            rank = token[:-1]
            suit_code = token[-1]
            suit_map = {"S": "spades", "C": "clubs", "D": "diamonds", "H": "hearts"}
            suit = suit_map.get(suit_code)
            if not suit or rank not in [c.rank.upper() for c in player_hand]:
                continue
            # find card in hand with matching rank and suit
            for c in player_hand:
                if c.rank.upper() == rank and c.suit == suit and c not in moves:
                    moves.append(c)
                    break
        return moves if moves else "PASS"

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
        play_history = game.move_history

        prompt = self.format_prompt(player_hand, play_history)

        api_key = GeminiAgent.get_next_api_key()
        key_str = api_key[:6] + "..." if api_key else "NONE"
        print("[GeminiAgent] >>> (Turn for AI player:", player.name, "| ID:", player.id, ")")
        print("[GeminiAgent] Prompt:\n" + prompt)
        print(f"[GeminiAgent] Using API key: {key_str}")

        if not api_key:
            print("[GeminiAgent] No API key available, passing move!")
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

        try:
            print("[GeminiAgent] Sending API request to Gemini...")
            resp = requests.post(api_url, json=payload, timeout=15, headers=headers)
            print(f"[GeminiAgent] Response HTTP status:", resp.status_code)
            print(f"[GeminiAgent] Raw response text:", resp.text)
            data = resp.json()
            llm_answer = ""
            if "candidates" in data and data["candidates"]:
                llm_answer = data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                llm_answer = data.get("text", "")
            print("[GeminiAgent] Parsed LLM answer:", llm_answer)
        except Exception as e:
            print(f"[GeminiAgent] ERROR during API call: {str(e)}")
            return "PASS"

        result = self.parse_llm_response(llm_answer, player_hand)
        print("[GeminiAgent] Final move selected:", result)
        return result