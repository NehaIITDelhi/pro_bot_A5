

import build.student_agent_module as student_agent
from typing import List, Dict, Any, Optional

# This is the wrapper class that 'agent.py' will import and use.
class StudentAgent:
    
    def __init__(self, player: str):
        """
        Initialize the C++ agent wrapper.
        'player' is "circle" or "square".
        """
        self.player = player
        # Create an instance of the *C++* StudentAgent class
        # from the compiled student_agent_module
        self.agent = student_agent.StudentAgent(player)

    def choose(self, py_board: List[List[Any]], rows: int, cols: int, score_cols: List[int], 
                 current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        This is the wrapper function called by the Python game engine.
        It translates Python objects to C++-compatible types, calls the C++
        agent, and translates the C++ result back to a Python dict.
        """
        
        # Import the C++ Piece type from the compiled module
        CppPiece = student_agent.Piece 
        
        # 1. TRANSLATE PYTHON BOARD TO C++ BOARD
        cpp_board = []
        for r in range(rows):
            row_list = []
            for c in range(cols):
                py_piece = py_board[r][c]
                
                if py_piece is None:
                    row_list.append(None)
                else:
                    # Create a C++ Piece object using the data from the Python Piece
                    # (The py_piece is an object from gameEngine.py)
                    cpp_piece_obj = CppPiece(py_piece.owner, py_piece.side, py_piece.orientation or "")
                    row_list.append(cpp_piece_obj)
            cpp_board.append(row_list)
        
        # 2. CALL THE C++ AGENT'S 'choose' FUNCTION
        # This calls the C++ code you wrote, passing the C++-compatible board
        cpp_move = self.agent.choose(cpp_board, rows, cols, score_cols, current_player_time, opponent_time)
        
        if cpp_move is None:
            return None

        # 3. TRANSLATE C++ MOVE OBJECT BACK TO PYTHON DICT
        
        # We map the C++ enum (e.g., ActionType.MOVE) to its string name
        action_map = {
            student_agent.ActionType.MOVE: "move",
            student_agent.ActionType.PUSH: "push",
            student_agent.ActionType.FLIP: "flip",
            student_agent.ActionType.ROTATE: "rotate",
        }
        action_str = action_map.get(cpp_move.action)

        # Use the 'from_pos' field we bound in C++
        move_dict = {
            "action": action_str,
            "from": cpp_move.from_pos,
        }
        
        # Add optional fields only if they are relevant
        if action_str in ("move", "push"):
            move_dict["to"] = cpp_move.to_pos
        
        if action_str == "push":
            move_dict["pushed_to"] = cpp_move.pushed_to
            
        if action_str == "flip" and cpp_move.orientation:
            move_dict["orientation"] = cpp_move.orientation

        return move_dict