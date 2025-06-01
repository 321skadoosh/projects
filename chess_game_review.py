import chess
import chess.pgn
import chess.engine
import io
import traceback
import asyncio


# Helper function for getting Centipawn value consistently from a specific color's perspective
def get_cp_value_for_color(score_obj, color):
    """
    Returns the centipawn value of a score from the perspective of 'color'.
    Handles both Score and PovScore objects.
    For mate scores, returns a very large positive/negative value for accurate loss calculation.
    """
    if score_obj is None:
        return 0  # No score available

    if score_obj.is_mate():
        mate_val = score_obj.mate()
        if mate_val is not None:
            # Return a very large centipawn value for mate scores
            mate_cp_value = 1000000
            if (color == chess.WHITE and mate_val > 0) or \
                    (color == chess.BLACK and mate_val < 0):  # Mate in favor of 'color'
                return mate_cp_value
            else:  # Mate against 'color'
                return -mate_cp_value
        return 0  # Should not happen if is_mate() is True and mate() is None

    if isinstance(score_obj, chess.engine.PovScore):
        # PovScore's white().cp or black().cp will give it from that specific color's perspective
        return score_obj.white().cp if color == chess.WHITE else score_obj.black().cp
    else:  # It's a chess.engine.Score object
        # Score.cp is typically from White's perspective. Negate if asking for Black.
        return score_obj.cp if color == chess.WHITE else -score_obj.cp


# Helper function to determine move rating based on centipawn loss
def get_move_rating(cp_loss: int, best_move_pv: list, player_move: chess.Move, is_mate_for_opponent: bool) -> str:
    """
    Determines the move rating based on centipawn loss, considering if it's a mate for the opponent.
    """
    # Define thresholds in centipawns for loss relative to the best move
    THRESHOLDS = {
        "Great": 15,  # Loss < 15 CP
        "Good": 30,  # Loss < 30 CP
        "Inaccuracy": 60,  # Loss < 60 CP
        "Mistake": 120,  # Loss < 120 CP
        "Blunder": 250,  # Loss < 250 CP
    }

    # If the player's move was exactly the engine's top choice
    if best_move_pv and player_move == best_move_pv[0]:
        return "Best Move"

    # If the move led to the opponent having a mate (i.e., this player blundered into mate)
    if is_mate_for_opponent:
        return "Blunder (Mate)"  # A more descriptive blunder rating

    # Otherwise, classify based on centipawn loss
    if cp_loss <= 5:  # Very minimal loss, but not the exact best move
        return "Optimal"
    elif cp_loss < THRESHOLDS["Great"]:
        return "Great"
    elif cp_loss < THRESHOLDS["Good"]:
        return "Good"
    elif cp_loss < THRESHOLDS["Inaccuracy"]:
        return "Inaccuracy"
    elif cp_loss < THRESHOLDS["Mistake"]:
        return "Mistake"
    elif cp_loss < THRESHOLDS["Blunder"]:
        return "Blunder"
    else:  # Very large loss
        return "Catastrophic Blunder"


async def review_chess_game(pgn_string: str, stockfish_path: str, time_limit: float = 0.1) -> list:
    game_review = []
    try:
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)

        if game is None:
            print("ERROR_FUNC: Could not parse the PGN string. Please ensure it's valid.")
            return []

        transport, engine = await chess.engine.popen_uci(stockfish_path)

        board = game.board()
        move_count = 0
        for move in game.mainline_moves():
            move_count += 1
            current_player_color = board.turn  # Store whose turn it is *before* the move

            # 1. Get the engine's ideal evaluation and best move for the *current* position (before player's move)
            # Request INFO_PV to get the principal variation (best move)
            analysis_before_move = await engine.analyse(
                board, chess.engine.Limit(time=time_limit), info=chess.engine.INFO_SCORE | chess.engine.INFO_PV
            )
            ideal_score_cp = get_cp_value_for_color(analysis_before_move.get("score"), current_player_color)
            best_move_pv = analysis_before_move.get("pv", [])  # Get the best move from the engine's perspective

            move_san = board.san(move)  # Get SAN before pushing the move
            board.push(move)  # Apply player's move to the board

            # 2. Get the engine's evaluation of the board *after* the player's move
            # Note: board.turn has now switched to the opponent
            analysis_after_move = await engine.analyse(
                board, chess.engine.Limit(time=time_limit), info=chess.engine.INFO_SCORE
            )

            # Get the score of the resulting position from the *current player's* (who just moved) perspective.
            # Since analysis_after_move.get("score") is from the *opponent's* new turn perspective, we negate it.
            actual_score_cp = -get_cp_value_for_color(analysis_after_move.get("score"), current_player_color)

            # 3. Calculate Centipawn Loss
            # Loss = (Ideal score for player) - (Actual score achieved by player's move)
            # We use max(0, ...) because loss cannot be negative.
            cp_loss = max(0, ideal_score_cp - actual_score_cp)

            # Determine if the move led to a mate for the opponent (a blunder)
            is_mate_for_opponent = analysis_after_move.get("score") and \
                                   analysis_after_move["score"].is_mate() and \
                                   analysis_after_move["score"].relative.mate() is not None and \
                                   analysis_after_move["score"].relative.mate() < 0

            move_rating = get_move_rating(cp_loss, best_move_pv, move, is_mate_for_opponent)

            # Prepare evaluation string (display the score from *after* the move)
            evaluation_str = "N/A"
            if analysis_after_move.get("score") is not None:
                score_obj_after_move = analysis_after_move["score"]  # This is the PovScore for the opponent

                if score_obj_after_move.is_mate():
                    mate_in = score_obj_after_move.relative.mate()
                    evaluation_str = f"M#{abs(mate_in)}" if mate_in is not None else "Mate"
                else:
                    # Always get the CP value from White's perspective for consistent display
                    cp_from_white_perspective = get_cp_value_for_color(score_obj_after_move, chess.WHITE)
                    evaluation_str = f"{cp_from_white_perspective / 100.0:.1f}"

            game_review.append({
                'move': move_san,
                'evaluation': evaluation_str,
                'rating': move_rating  # Add the new rating
            })

        await engine.quit()

    except FileNotFoundError:
        print(f"ERROR_FUNC: Stockfish executable not found at '{stockfish_path}'. "
              "Please check the path and ensure Stockfish is installed and accessible.")
        return []
    except chess.engine.EngineError as ee:
        print(
            f"ERROR_FUNC: Chess engine error: {ee}. This might indicate a problem with Stockfish or its communication.")
        traceback.print_exc()
        return []
    except Exception as e:
        traceback.print_exc()
        return []

    return game_review


if __name__ == "__main__":
    # Get PGN input from the user
    print("Please paste your PGN game string below.")
    print("When you are finished, type 'END_PGN' on a new line and press Enter.")
    pgn_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END_PGN":
            break
        pgn_lines.append(line)
    user_pgn_string = "\n".join(pgn_lines)

    # --- STOCKFISH PATH CONFIGURATION ---
    # IMPORTANT: Replace this with the actual path to your Stockfish executable.
    stockfish_exe_path = r"filepath"

    if not user_pgn_string.strip():
        print("No PGN was entered. Exiting.")
    else:
        try:
            game_analysis = asyncio.run(review_chess_game(user_pgn_string, stockfish_exe_path))
        except Exception as e:
            print(f"MAIN_ERROR: An error occurred during game review: {e}")
            traceback.print_exc()
            game_analysis = []

        if game_analysis:
            print("\n--- Game Review ---")
            for i, move_data in enumerate(game_analysis):
                move_number = (i // 2) + 1
                player_move = "White" if i % 2 == 0 else "Black"
                print(
                    f"Move {move_number}. {player_move}: {move_data['move']} | Evaluation: {move_data['evaluation']} | Rating: {move_data['rating']}")
        else:
            print("Game review could not be completed. Check for errors above.")
