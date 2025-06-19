import sys
sys.path.append(".")  # Make sure src/ (project root) imports work

from src.ai.agent import AIAgent
from src.game.card import Card
from src.game.player import Player
import unittest

class TestAgentActions(unittest.TestCase):
    def setUp(self):
        class DummyGame:
            pass
        self.DummyGame = DummyGame
        self.agent = AIAgent(0)
        
    def create_player(self, cards):
        player = Player("Test", "bottom", is_ai=True)
        player.hand = [Card(suit, rank) for suit, rank in cards]
        return player
        
    def create_game_state(self, player, last_played=None):
        game_state = self.DummyGame()
        game_state.players = [player]
        game_state.last_played_cards = last_played if last_played else []
        return game_state

    def test_first_turn_with_3spades(self):
        """Test first turn with 3♠ - must play 3♠ only"""
        player = self.create_player([
            ('spades', '3'), ('hearts', '4'), ('clubs', '5')
        ])
        game_state = self.create_game_state(player)
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should only have actions for 3♠
        self.assertEqual(len(actions), 1)
        self.assertTrue(actions[0] > 0 and actions[0] <= 52)
        print("✓ First turn with 3♠: Only 3♠ is valid")

    def test_first_turn_no_3spades(self):
        """Test first turn without 3♠ - must PASS only"""
        player = self.create_player([
            ('hearts', '4'), ('clubs', '5'), ('diamonds', '6')
        ])
        game_state = self.create_game_state(player)
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should only have PASS action
        self.assertEqual(actions, [0])
        print("✓ First turn without 3♠: Only PASS is valid")

    def test_beat_single_card(self):
        """Test beating a single card"""
        player = self.create_player([
            ('spades', '5'), ('hearts', '7'), ('diamonds', 'K')
        ])
        game_state = self.create_game_state(
            player, 
            last_played=[Card('clubs', '6')]
        )
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should have PASS + all cards higher than 6
        self.assertIn(0, actions)
        self.assertGreater(len(actions), 1)
        print(f"✓ Beat single card: Found {len(actions)} valid actions")

    def test_beat_pair(self):
        """Test beating a pair"""
        player = self.create_player([
            ('spades', '5'), ('hearts', '5'),
            ('diamonds', '7'), ('clubs', '7'),
            ('spades', '8'), ('hearts', '8')
        ])
        game_state = self.create_game_state(
            player, 
            last_played=[Card('diamonds', '4'), Card('clubs', '4')]
        )
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should have PASS + pairs higher than 4
        self.assertIn(0, actions)
        self.assertGreater(len(actions), 1)
        print(f"✓ Beat pair: Found {len(actions)} valid actions")

    def test_beat_triple(self):
        """Test beating a triple"""
        player = self.create_player([
            ('spades', '6'), ('hearts', '6'), ('diamonds', '6'),
            ('clubs', '8'), ('spades', '8'), ('hearts', '8')
        ])
        game_state = self.create_game_state(
            player, 
            last_played=[
                Card('diamonds', '5'), 
                Card('clubs', '5'), 
                Card('hearts', '5')
            ]
        )
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should have PASS + triples higher than 5
        self.assertIn(0, actions)
        self.assertGreater(len(actions), 1)
        print(f"✓ Beat triple: Found {len(actions)} valid actions")

    def test_beat_straight(self):
        """Test beating a straight"""
        player = self.create_player([
            ('spades', '4'), ('hearts', '5'), ('diamonds', '6'),
            ('clubs', '7'), ('spades', '8'), ('hearts', '9'),
            ('diamonds', '10'), ('clubs', 'J')
        ])
        game_state = self.create_game_state(
            player, 
            last_played=[
                Card('clubs', '3'), 
                Card('diamonds', '4'), 
                Card('hearts', '5')
            ]
        )
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should have PASS + straights of same length or longer
        self.assertIn(0, actions)
        self.assertGreater(len(actions), 1)
        print(f"✓ Beat straight: Found {len(actions)} valid actions")

    def test_beat_four_of_a_kind(self):
        """Test beating four of a kind"""
        player = self.create_player([
            ('spades', '9'), ('hearts', '9'), ('diamonds', '9'), ('clubs', '9'),
            ('spades', 'A'), ('hearts', 'A'), ('diamonds', 'A'), ('clubs', 'A')
        ])
        game_state = self.create_game_state(
            player, 
            last_played=[
                Card('spades', '8'), 
                Card('hearts', '8'), 
                Card('diamonds', '8'), 
                Card('clubs', '8')
            ]
        )
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should have PASS + four of a kind higher than 8
        self.assertIn(0, actions)
        self.assertGreater(len(actions), 1)
        print(f"✓ Beat four of a kind: Found {len(actions)} valid actions")

    def test_beat_pair_straight(self):
        """Test beating a pair straight"""
        player = self.create_player([
            ('spades', '7'), ('hearts', '7'),
            ('diamonds', '8'), ('clubs', '8'),
            ('spades', '9'), ('hearts', '9'),
            ('diamonds', '10'), ('clubs', '10')
        ])
        game_state = self.create_game_state(
            player, 
            last_played=[
                Card('diamonds', '4'), Card('clubs', '4'),
                Card('spades', '5'), Card('hearts', '5'),
                Card('diamonds', '6'), Card('clubs', '6')
            ]
        )
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should have PASS + pair straights of same length or longer
        self.assertIn(0, actions)
        self.assertGreater(len(actions), 1)
        print(f"✓ Beat pair straight: Found {len(actions)} valid actions")

    def test_no_valid_actions(self):
        """Test when no valid moves - must PASS"""
        player = self.create_player([
            ('spades', '3'), ('hearts', '4'), ('clubs', '5')
        ])
        game_state = self.create_game_state(
            player, 
            last_played=[Card('diamonds', 'K')]
        )
        
        actions = self.agent.get_valid_actions(game_state)
        
        # Should only have PASS
        self.assertEqual(actions, [0])
        print("✓ No valid moves: Only PASS is valid")

    def test_complex_combinations(self):
        """Test complex hand with all combination types"""
        player = self.create_player([
            ('spades', '3'), ('spades', '4'), ('hearts', '3'), ('hearts', '4'),
            ('spades', '5'), ('diamonds', '5'), ('clubs', '5'), ('hearts', '5'),
            ('spades', '6'), ('hearts', '6'),
            ('spades', '7'), ('hearts', '7'), ('clubs', '4')
        ])
        game_state = self.create_game_state(player)
        
        # First turn actions
        actions1 = self.agent.get_valid_actions(game_state)
        self.assertEqual(len(actions1), 1)  # Must play 3♠
        print("✓ Complex hand first turn: Only 3♠ valid")
        
        # Test beating a pair
        game_state.last_played_cards = [
            Card('spades', '4'), Card('clubs', '4')
        ]
        actions2 = self.agent.get_valid_actions(game_state)
        self.assertGreater(len(actions2), 1)
        print(f"✓ Complex hand beat pair: {len(actions2)} valid actions")
        
        # Test beating a straight
        game_state.last_played_cards = [
            Card('spades', '3'), Card('spades', '4'), 
            Card('spades', '5'), Card('spades', '6'), 
            Card('spades', '7')
        ]
        actions3 = self.agent.get_valid_actions(game_state)
        self.assertGreater(len(actions3), 1)
        print(f"✓ Complex hand beat straight: {len(actions3)} valid actions")

if __name__ == "__main__":
    # Run all test cases
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgentActions)
    unittest.TextTestRunner(verbosity=2).run(suite)