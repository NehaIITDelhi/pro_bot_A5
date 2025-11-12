#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // This handles optional, vector, and map
#include "student_agent_improved.h" // <-- Make sure this matches your .h file name
#include <sstream> // For string building
#include <set>
#include <random>
#include <fstream> // For file I/O
#include <deque>
#include <algorithm>
#include <limits> // For infinity

namespace py = pybind11;

// ===================================================================
// NEW: DEFINE STATIC HISTORY DEQUE
// ===================================================================
std::deque<uint64_t> StudentAgent::recent_positions;

// ===================================================================
// CONSTRUCTOR - NOW INITS ZOBRIST
// ===================================================================
StudentAgent::StudentAgent(std::string p) : BaseAgent(p), turn_count(0), random_engine(std::random_device{}()) {
    // We only need to init Zobrist once
    init_zobrist(20, 20); // Init for max board size
}

// ===================================================================
// ZOBRIST HASHING FUNCTIONS
// ===================================================================
void StudentAgent::init_zobrist(int rows, int cols) {
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    
    zobrist_table.resize(NUM_PIECE_TYPES, std::vector<uint64_t>(rows * cols));
    for (int i = 0; i < NUM_PIECE_TYPES; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            zobrist_table[i][j] = dist(random_engine);
        }
    }
}

int StudentAgent::get_zobrist_index(const PiecePtr& piece) const {
    if (!piece) return -1; // Should not happen
    if (piece->owner == "circle") {
        if (piece->side == "stone") return 0;
        if (piece->orientation == "horizontal") return 1;
        return 2; // Vertical
    } else {
        if (piece->side == "stone") return 3;
        if (piece->orientation == "horizontal") return 4;
        return 5; // Vertical
    }
}

uint64_t StudentAgent::board_hash_zobrist(const Board& board, int rows, int cols) const {
    uint64_t hash = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (board[r][c]) {
                int piece_index = get_zobrist_index(board[r][c]);
                hash ^= zobrist_table[piece_index][r * cols + c];
            }
        }
    }
    return hash;
}

// ===================================================================
// UTILITY FUNCTIONS (Unchanged)
// ===================================================================

bool StudentAgent::in_bounds(int x, int y, int rows, int cols) const {
    return 0 <= x && x < cols && 0 <= y && y < rows;
}
std::vector<int> StudentAgent::score_cols_for(int cols) const {
    int w = 4; int start = std::max(0, (cols - w) / 2);
    std::vector<int> result; for (int i = 0; i < w; ++i) result.push_back(start + i);
    return result;
}
int StudentAgent::top_score_row() const { return 2; }
int StudentAgent::bottom_score_row(int rows) const { return rows - 3; }
bool StudentAgent::is_opponent_score_cell(int x, int y, const std::string& p, int rows, int cols, const std::vector<int>& score_cols) const {
    int opp_row = (p == "circle") ? bottom_score_row(rows) : top_score_row();
    if (y != opp_row) return false;
    for (int col : score_cols) if (x == col) return true;
    return false;
}
bool StudentAgent::is_own_score_cell(int x, int y, const std::string& p, int rows, int cols, const std::vector<int>& score_cols) const {
    int own_row = (p == "circle") ? top_score_row() : bottom_score_row(rows);
    if (y != own_row) return false;
    for (int col : score_cols) if (x == col) return true;
    return false;
}
// size_t StudentAgent::board_hash(const Board& board) const { // <-- REMOVED slow hash
// ...
// }
int StudentAgent::manhattan_distance(int x1, int y1, int x2, int y2) const {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}
Board StudentAgent::deep_copy_board(const Board& board) {
    Board new_board; new_board.resize(board.size());
    for (size_t r = 0; r < board.size(); ++r) {
        new_board[r].resize(board[r].size());
        for (size_t c = 0; c < board[r].size(); ++c) {
            if (board[r][c]) new_board[r][c] = std::make_shared<Piece>(*board[r][c]);
            else new_board[r][c] = nullptr;
        }
    }
    return new_board;
}

// ===================================================================
// MODIFIED: MAIN SEARCH FUNCTION
// ===================================================================

std::optional<Move> StudentAgent::choose(
    const Board& board, int rows, int cols, const std::vector<int>& score_cols,
    double current_player_time, double opponent_time) 
{
    start_time_point = std::chrono::high_resolution_clock::now();
    turn_count++; 
    transposition_table.clear();

    uint64_t current_zobrist_hash = board_hash_zobrist(board, rows, cols);

    // --- SHARED HISTORY LOGIC (Unchanged) ---
    auto is_starting_board = [&](const Board& b) {
        if (this->player != "circle") return false;
        int piece_count = 0;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (b[r][c]) piece_count++;
            }
        }
        return (piece_count == 24 || piece_count == 28 || piece_count == 32);
    };

    if (is_starting_board(board)) {
        recent_positions.clear();
    }
    
    recent_positions.push_back(current_zobrist_hash); 
    if (recent_positions.size() > MAX_HISTORY_SIZE) {
        recent_positions.pop_front();
    }
    // --- END SHARED HISTORY ---

    // --- MODIFIED: AGGRESSIVE ADAPTIVE TIME MANAGEMENT ---
    if (current_player_time > 45.0) time_limit_seconds = 2.0; 
    else if (current_player_time > 25.0) time_limit_seconds = 1.2;
    else if (current_player_time > 10.0) time_limit_seconds = 0.8; 
    else time_limit_seconds = std::max(0.1, current_player_time / 15.0); 
    // --- END MODIFIED TIME ---

    auto moves = get_all_valid_moves_enhanced(board, this->player, rows, cols, score_cols);
    if (moves.empty()) return std::nullopt;

    // (Immediate win/threat checks remain the same)
    for (const auto& move : moves) {
        if (is_winning_move(board, move, this->player, rows, cols, score_cols)) {
            update_move_history(move); return move;
        }
    }

    auto opp_moves = get_all_valid_moves_enhanced(board, this->opponent, rows, cols, score_cols);
    for (const auto& opp_move : opp_moves) {
        if (is_winning_move(board, opp_move, this->opponent, rows, cols, score_cols)) {
            auto defensive_move = find_blocking_move(board, moves, opp_move, rows, cols, score_cols);
            if (defensive_move) {
                update_move_history(*defensive_move); return *defensive_move;
            }
        }
    }
    
    std::string game_phase = get_game_phase(board, rows, cols, score_cols);
    int max_depth;
    if (current_player_time < 5) max_depth = 1;
    else if (current_player_time < 15) max_depth = 2;
    else if (game_phase == "endgame") max_depth = 4;
    else if (game_phase == "midgame") max_depth = 3;
    else max_depth = 2;

    moves = order_moves_with_edge_control(board, moves, this->player, rows, cols, score_cols);

    // --- MODIFIED: WIDER SEARCH AT ROOT FOR AMPLE TIME ---
    int max_moves;
    if (current_player_time > 20) max_moves = 15;
    else max_moves = 8;
    if (moves.size() > max_moves) moves.resize(max_moves);
    // --- END MODIFIED MOVE COUNT ---

    Board board_copy = deep_copy_board(board);
    
    Move best_move = moves[0];
    double best_score = -std::numeric_limits<double>::infinity();

    for (int depth = 1; depth <= max_depth; ++depth) {
        auto now = std::chrono::high_resolution_clock::now();
        double time_spent = std::chrono::duration<double>(now - start_time_point).count();
        if (time_spent > time_limit_seconds * 0.85) break;

        double alpha = -std::numeric_limits<double>::infinity();
        double beta = std::numeric_limits<double>::infinity();
        Move current_best;

        for (int i = 0; i < moves.size(); ++i) {
            
            // --- CRITICAL TIME CHECK FOR ITERATIVE DEEPENING ---
            now = std::chrono::high_resolution_clock::now();
            time_spent = std::chrono::duration<double>(now - start_time_point).count();
            if (time_spent > time_limit_seconds) break;
            // --- END TIME CHECK ---

            UndoInfo undo_data = apply_move_inplace(board_copy, moves[i], rows, cols, current_zobrist_hash);
            
            double score;
            if (i == 0) {
                 score = -negamax_with_balance(board_copy, depth - 1, -beta, -alpha, 
                                               this->opponent, rows, cols, score_cols, depth, current_zobrist_hash);
            } else {
                score = -negamax_with_balance(board_copy, depth - 1, -alpha - 1, -alpha,
                                              this->opponent, rows, cols, score_cols, depth, current_zobrist_hash);
                if (alpha < score && score < beta) {
                    score = -negamax_with_balance(board_copy, depth - 1, -beta, -score,
                                                  this->opponent, rows, cols, score_cols, depth, current_zobrist_hash);
                }
            }

            unapply_move(board_copy, moves[i], undo_data, rows, cols, current_zobrist_hash);

            if (score > best_score) {
                best_score = score;
                current_best = moves[i];
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) break;
        }
        
        if (current_best.action != Move::ActionType::MOVE || current_best.from != std::make_pair(0,0)) {
             best_move = current_best;
             auto it = std::find(moves.begin(), moves.end(), best_move);
             if (it != moves.end()) {
                 moves.erase(it);
                 moves.insert(moves.begin(), best_move);
             }
        }
    }

    if (best_score < -9000 && !moves.empty()) {
        std::shuffle(moves.begin(), moves.end(), random_engine);
        best_move = moves[0];
    }

    update_move_history(best_move);
    return best_move;
}

void StudentAgent::update_move_history(const Move& move) {
    last_moves.push_back(move);
    if (last_moves.size() > MAX_RECENT_MOVES) last_moves.pop_front();
}
bool StudentAgent::moves_similar(const Move& move1, const Move& move2) const {
    if (move1.action != move2.action) return false;
    if (move1.from == move2.from && move1.to == move2.to) return true;
    if (move1.action == Move::ActionType::MOVE && move2.action == Move::ActionType::MOVE) {
        if (move1.from == move2.to && move1.to == move2.from) return true;
    }
    return false;
}

// ===================================================================
// MODIFIED: NEGAMAX IMPLEMENTATION (WITH FAST REPETITION PENALTY)
// ===================================================================

double StudentAgent::negamax_with_balance(Board& board, int depth, double alpha, double beta,
                                          const std::string& current_player, int rows, int cols, 
                                          const std::vector<int>& score_cols, int max_depth, 
                                          uint64_t& zobrist_hash) 
{
    double original_alpha = alpha;
    
    // --- Transposition Table Lookup ---
    if (auto it = transposition_table.find(zobrist_hash); it != transposition_table.end()) {
        TTEntry& entry = it->second;
        if (entry.depth >= depth) {
            if (entry.flag == TTFlag::EXACT) return entry.score;
            if (entry.flag == TTFlag::LOWER_BOUND) alpha = std::max(alpha, entry.score);
            else if (entry.flag == TTFlag::UPPER_BOUND) beta = std::min(beta, entry.score);
            if (alpha >= beta) return entry.score;
        }
    }

    auto now = std::chrono::high_resolution_clock::now();
    double time_spent = std::chrono::duration<double>(now - start_time_point).count();
    if (time_spent > time_limit_seconds) {
        return evaluate_balanced(board, current_player, rows, cols, score_cols);
    }

    // --- Dynamic Win/Loss Check (Terminal Node) ---
    size_t required_win_count = score_cols.size();
    
    int my_stones = count_stones_in_score_area(board, current_player, rows, cols, score_cols);
    std::string opp = get_opponent(current_player);
    int opp_stones = count_stones_in_score_area(board, opp, rows, cols, score_cols);
    
    // DYNAMIC WIN CHECK: >= score_cols.size() (4, 5, or 6)
    if (my_stones >= required_win_count) return 10000.0 - (max_depth - depth) * 10.0;
    if (opp_stones >= required_win_count) return -10000.0 + (max_depth - depth) * 10.0;

    // --- Quiescence Search ---
    if (depth <= 0) {
        return quiescence_search(board, alpha, beta, current_player, rows, cols, score_cols, zobrist_hash, 2); // Max q-depth of 2
    }

    auto moves = get_all_valid_moves_enhanced(board, current_player, rows, cols, score_cols);
    if (moves.empty()) {
        return evaluate_balanced(board, current_player, rows, cols, score_cols);
    }
    
    moves = order_moves_with_edge_control(board, moves, current_player, rows, cols, score_cols);

    int max_moves = (depth <= 1) ? 4 : 6;
    if (moves.size() > max_moves) moves.resize(max_moves);

    double best_score = -std::numeric_limits<double>::infinity();

    for (const auto& move : moves) {
        UndoInfo undo_data = apply_move_inplace(board, move, rows, cols, zobrist_hash);
        
        double score;
        
        // --- NEW: REPETITION PENALTY LOGIC (Using Zobrist) ---
        bool found_in_history = false;
        
        // Check for immediate 2-ply repetition (A->B->A)
        if (recent_positions.size() >= 2 && zobrist_hash == recent_positions[recent_positions.size() - 2]) {
            score = -9999; // Heavy penalty
            found_in_history = true;
        } else {
            // Check if it's any other recent position
            for (const uint64_t& hash : recent_positions) { // <-- Use Zobrist hash type
                if (zobrist_hash == hash) {
                    score = -5000; // Milder penalty
                    found_in_history = true;
                    break;
                }
            }
        }
        
        if (!found_in_history) {
            // Not a loop, proceed with normal search
            score = -negamax_with_balance(board, depth - 1, -beta, -alpha,
                                          opp, rows, cols, score_cols, max_depth, zobrist_hash);
        }
        // --- END NEW ---
        
        unapply_move(board, move, undo_data, rows, cols, zobrist_hash);

        if (score > best_score) best_score = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            if (killer_moves.find(depth) == killer_moves.end()) killer_moves[depth] = {};
            auto& killers = killer_moves[depth];
            if (std::find(killers.begin(), killers.end(), move) == killers.end()) {
                killers.insert(killers.begin(), move);
                if (killers.size() > 2) killers.pop_back();
            }
            break;
        }
    }

    // --- Transposition Table Store ---
    TTEntry entry = {best_score, depth, TTFlag::EXACT};
    if (best_score <= original_alpha) entry.flag = TTFlag::UPPER_BOUND;
    else if (best_score >= beta) entry.flag = TTFlag::LOWER_BOUND;
    transposition_table[zobrist_hash] = entry;
    
    return best_score;
}

// ===================================================================
// QUIESCENCE SEARCH (Unchanged)
// ===================================================================

double StudentAgent::quiescence_search(Board& board, double alpha, double beta,
                                       const std::string& current_player, int rows, int cols,
                                       const std::vector<int>& score_cols, uint64_t& zobrist_hash, int q_depth) 
{
    if (auto it = transposition_table.find(zobrist_hash); it != transposition_table.end()) {
        TTEntry& entry = it->second;
        if (entry.flag == TTFlag::EXACT) return entry.score;
        if (entry.flag == TTFlag::LOWER_BOUND) alpha = std::max(alpha, entry.score);
        else if (entry.flag == TTFlag::UPPER_BOUND) beta = std::min(beta, entry.score);
        if (alpha >= beta) return entry.score;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    double time_spent = std::chrono::duration<double>(now - start_time_point).count();
    if (time_spent > time_limit_seconds) {
        return evaluate_balanced(board, current_player, rows, cols, score_cols);
    }
    
    double best_score = evaluate_balanced(board, current_player, rows, cols, score_cols);
    
    if (q_depth <= 0) {
        return best_score;
    }
    if (best_score >= beta) {
        return best_score; // Fail-high
    }
    alpha = std::max(alpha, best_score);
    
    auto all_moves = get_all_valid_moves_enhanced(board, current_player, rows, cols, score_cols);
    std::vector<Move> forcing_moves;
    for (const auto& move : all_moves) {
        if (get_move_priority(board, move, current_player, rows, cols, score_cols) > 1500) {
            forcing_moves.push_back(move);
        }
    }
    
    if (forcing_moves.empty()) {
        return best_score;
    }
    
    for (const auto& move : forcing_moves) {
        UndoInfo undo_data = apply_move_inplace(board, move, rows, cols, zobrist_hash);
        
        std::string opp = get_opponent(current_player);
        double score = -quiescence_search(board, -beta, -alpha,
                                          opp, rows, cols, score_cols, zobrist_hash, q_depth - 1);
        
        unapply_move(board, move, undo_data, rows, cols, zobrist_hash);

        if (score > best_score) best_score = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    
    return best_score;
}


// ===================================================================
// PHASE-AWARE EVALUATION (Unchanged)
// ===================================================================

double StudentAgent::evaluate_balanced(const Board& board, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols) {
    
    std::string game_phase = get_game_phase(board, rows, cols, score_cols);
    double edge_control_weight = 100.0, defensive_pos_weight = 80.0, manhattan_dist_weight = 6.0;
    double ready_to_score_weight = 150.0, balance_weight = 30.0, river_network_weight = 50.0;

    // --- MODIFIED ENDGAME WEIGHTS ---
    if (game_phase == "endgame") {
        ready_to_score_weight = 1500.0; // Greatly increased to prioritize scoring immediately
        manhattan_dist_weight = 20.0;
        edge_control_weight = 10.0; 
        defensive_pos_weight = 20.0;
    } else if (game_phase == "opening") {
        defensive_pos_weight = 120.0; 
        edge_control_weight = 130.0;
        balance_weight = 50.0; 
        ready_to_score_weight = 75.0;
    }
    // --- END MODIFIED WEIGHTS ---
    
    double score = 0.0; 
    std::string opp = get_opponent(current_player);
    int my_scored = count_stones_in_score_area(board, current_player, rows, cols, score_cols);
    int opp_scored = count_stones_in_score_area(board, opp, rows, cols, score_cols);
    
    // --- Dynamic Win/Loss Terminals and Near-Win Bonuses ---
    size_t required_win_count = score_cols.size();
    
    // Win/Loss Terminals
    score += my_scored * 1000.0; 
    score -= opp_scored * 1100.0;
    
    if (my_scored >= required_win_count) return 10000.0; 
    if (opp_scored >= required_win_count) return -10000.0;
    
    // Near-Win Bonuses
    if (my_scored == required_win_count - 1) score += 500.0; 
    if (opp_scored == required_win_count - 1) score -= 600.0;

    // Weighted Feature Sum
    score += evaluate_edge_control(board, current_player, rows, cols, score_cols) * edge_control_weight;
    score += evaluate_defensive_position(board, current_player, rows, cols, score_cols) * defensive_pos_weight;
    score += evaluate_manhattan_distances(board, current_player, rows, cols, score_cols) * manhattan_dist_weight;
    score -= evaluate_manhattan_distances(board, opp, rows, cols, score_cols) * manhattan_dist_weight;
    score += evaluate_river_network_balanced(board, current_player, rows, cols, score_cols) * river_network_weight;
    score -= evaluate_river_network_balanced(board, opp, rows, cols, score_cols) * river_network_weight;
    score += count_stones_ready_to_score(board, current_player, rows, cols, score_cols) * ready_to_score_weight;
    score -= count_stones_ready_to_score(board, opp, rows, cols, score_cols) * ready_to_score_weight;
    score += evaluate_balance_factor(board, current_player, rows, cols) * balance_weight;
    
    return score;
}

// --- (Evaluation helper functions are unchanged) ---
double StudentAgent::evaluate_edge_control(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double edge_score = 0.0; int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    std::vector<int> edge_cols = {0, 1, cols - 2, cols - 1};
    for (int col : edge_cols) {
        int start_row = (player == "circle") ? (target_row + 1) : std::max(target_row - 3, 0);
        int end_row = (player == "circle") ? std::min(target_row + 4, rows) : target_row;
        for (int row = start_row; row < end_row; ++row) {
            if (in_bounds(col, row, rows, cols)) {
                auto piece = board[row][col];
                if (piece) {
                    if (piece->owner == player) {
                        edge_score += 1.5;
                        if (piece->side == "river" && piece->orientation == "horizontal") edge_score += 2.0;
                    } else edge_score -= 2.0;
                }
            }
        }
    }
    for (int col : {0, cols - 1}) {
        bool lane_blocked = false;
        int start_row = (player == "circle") ? (target_row + 1) : (top_score_row() + 1);
        int end_row = (player == "circle") ? bottom_score_row(rows) : target_row;
        for (int row = start_row; row < end_row; ++row) {
            if (in_bounds(col, row, rows, cols) && board[row][col]) { lane_blocked = true; break; }
        }
        if (!lane_blocked) edge_score -= 3.0;
    }
    return edge_score;
}
double StudentAgent::evaluate_defensive_position(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double defensive_score = 0.0; std::string opp = get_opponent(player);
    int start_row = (player == "circle") ? (top_score_row() + 1) : std::max(bottom_score_row(rows) - 3, 0);
    int end_row = (player == "circle") ? std::min(top_score_row() + 4, rows) : bottom_score_row(rows);
    std::vector<bool> col_coverage(cols, false);
    for (int row = start_row; row < end_row; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (in_bounds(col, row, rows, cols)) {
                auto piece = board[row][col];
                if (piece && piece->owner == player) {
                    col_coverage[col] = true; defensive_score += 1.0;
                    if (std::find(score_cols.begin(), score_cols.end(), col) != score_cols.end() || col == 0 || col == cols - 1) defensive_score += 1.5;
                    if (piece->side == "river") defensive_score += 0.5;
                }
            }
        }
    }
    for (int col = 0; col < cols; ++col) {
        if (!col_coverage[col]) {
            if (std::find(score_cols.begin(), score_cols.end(), col) != score_cols.end()) defensive_score -= 2.0;
            else if (col == 0 || col == cols - 1) defensive_score -= 1.5;
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            auto piece = board[row][col];
            if (piece && piece->owner == opp && piece->side == "stone") {
                if (player == "circle" && row >= bottom_score_row(rows) - 2) defensive_score -= 3.0;
                else if (player == "square" && row <= top_score_row() + 2) defensive_score -= 3.0;
            }
        }
    }
    return defensive_score;
}
double StudentAgent::evaluate_balance_factor(const Board& board, const std::string& player, int rows, int cols) {
    double balance_score = 0.0; int offensive_pieces = 0, defensive_pieces = 0, mid_pieces = 0; int mid_row = rows / 2;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            auto piece = board[row][col];
            if (piece && piece->owner == player) {
                if (player == "circle") {
                    if (row < mid_row - 1) offensive_pieces++;
                    else if (row > mid_row + 1) defensive_pieces++;
                    else mid_pieces++;
                } else {
                    if (row > mid_row + 1) offensive_pieces++;
                    else if (row < mid_row - 1) defensive_pieces++;
                    else mid_pieces++;
                }
            }
        }
    }
    int total = offensive_pieces + defensive_pieces + mid_pieces;
    if (total > 0) {
        double off_ratio = (double)offensive_pieces / total; double def_ratio = (double)defensive_pieces / total; double mid_ratio = (double)mid_pieces / total;
        if (off_ratio > 0.7) balance_score -= 5.0;
        else if (def_ratio > 0.6) balance_score -= 3.0;
        else if (off_ratio < 0.2) balance_score -= 4.0;
        else balance_score += 3.0;
        balance_score += mid_ratio * 5.0;
    }
    return balance_score;
}
double StudentAgent::evaluate_river_network_balanced(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double score = 0.0; std::vector<std::tuple<int, int, PiecePtr>> rivers;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            auto p = board[r][c]; if (p && p->owner == player && p->side == "river") rivers.emplace_back(r, c, p);
        }
    }
    for (size_t i = 0; i < rivers.size(); ++i) {
        auto [r1, c1, p1] = rivers[i];
        for (size_t j = i + 1; j < rivers.size(); ++j) {
            auto [r2, c2, p2] = rivers[j];
            if (std::abs(r1 - r2) + std::abs(c1 - c2) <= 3) { score += 1.0; if (p1->orientation != p2->orientation) score += 0.5; }
        }
    }
    int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    for (auto [r, c, p] : rivers) {
        if (player == "circle") {
            if (r > target_row && r < target_row + 3) { score += 2.0; if (p->orientation == "horizontal") score += 1.0; }
        } else {
            if (r < target_row && r > target_row - 3) { score += 2.0; if (p->orientation == "horizontal") score += 1.0; }
        }
        if (c == 0 || c == 1 || c == cols - 2 || c == cols - 1) score += 1.5;
    }
    return score;
}
double StudentAgent::evaluate_manhattan_distances(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double total_score = 0.0; 
    int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            auto piece = board[row][col];
            if (piece && piece->owner == player && piece->side == "stone") {
                if (is_own_score_cell(col, row, player, rows, cols, score_cols)) continue;
                
                int min_dist = std::numeric_limits<int>::max();
                for (int score_col : score_cols) 
                    min_dist = std::min(min_dist, manhattan_distance(col, row, score_col, target_row));
                
                // --- MODIFIED: AGGRESSIVE PROXIMITY REWARD ---
                if (min_dist == 1) {
                    total_score += 100.0; // Large bonus for being one step away
                } else if (min_dist > 0) {
                    // Standard logic with slightly sharper decay
                    total_score += std::max(0, 25 - min_dist * 3);
                }
                // --- END MODIFIED PROXIMITY ---
                
                if (std::find(score_cols.begin(), score_cols.end(), col) != score_cols.end()) total_score += 8.0;
                int row_dist = std::abs(row - target_row); 
                if (row_dist <= 2) total_score += (3 - row_dist) * 4.0;
            }
        }
    }
    return total_score;
}
double StudentAgent::count_stones_ready_to_score(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double count = 0.0; int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    int check_row = (player == "circle") ? (target_row + 1) : (target_row - 1);
    if (check_row >= 0 && check_row < rows) {
        for (int col : score_cols) {
            if (in_bounds(col, check_row, rows, cols)) {
                auto piece = board[check_row][col];
                if (piece && piece->owner == player && piece->side == "stone") {
                    if (!board[target_row][col]) count += 1.0;
                }
            }
        }
    }
    return count;
}

// ===================================================================
// MOVE ORDERING (Refactored)
// ===================================================================

int StudentAgent::get_move_priority(const Board& board, const Move& move, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols) {
    int priority = 0;
    
    // 1. Immediate Win Check (Absolute Highest Priority)
    if (is_winning_move(board, move, current_player, rows, cols, score_cols)) return 100000;

    // 2. Killer Moves (High Priority)
    if (killer_moves.count(turn_count)) {
        const auto& killers = killer_moves[turn_count];
        if (std::find(killers.begin(), killers.end(), move) != killers.end()) {
            priority += 5000;
        }
    }

    // --- MODIFIED PUSH LOGIC (PRIORITY BOOST) ---
    if (move.action == Move::ActionType::PUSH) {
        auto from_piece = board[move.from.second][move.from.first];
        auto pushed_piece = board[move.to.second][move.to.first];
        
        if (from_piece && pushed_piece) {
            // Push opponent piece (HIGH DEFENSIVE/OFFENSIVE VALUE)
            if (pushed_piece->owner == get_opponent(current_player)) { 
                priority += 3000; 

                // River push of opponent stone (MOST POWERFUL)
                if (from_piece->side == "river") {
                    priority += 1500; 
                }
            }
            // Push own stone closer to score (HIGH OFFENSIVE VALUE)
            else { 
                int target_row = (current_player == "circle") ? top_score_row() : bottom_score_row(rows);
                int old_dist = std::abs(move.to.second - target_row);
                int new_dist = std::abs(move.pushed_to.second - target_row);
                
                if (new_dist < old_dist) {
                    priority += 2500;
                }
            }
        }
    }
    // --- END MODIFIED PUSH LOGIC ---

    // 3. Simple Move into Score Area (High Value)
    if (move.action == Move::ActionType::MOVE) {
        if (is_own_score_cell(move.to.first, move.to.second, current_player, rows, cols, score_cols)) {
            priority += 4000;
        }
    }

    // 4. Simple Move Advancing Closer to Score (Mid-High Value)
    if (move.action == Move::ActionType::MOVE) {
        int target_row = (current_player == "circle") ? top_score_row() : bottom_score_row(rows);
        int old_dist = std::abs(move.from.second - target_row); 
        int new_dist = std::abs(move.to.second - target_row);
        
        // Use a better scaling factor for distance reduction
        if (new_dist < old_dist) priority += (old_dist - new_dist) * 300; 
    }
    
    // 5. Edge Control / Flips / Rotation (Lower but important factors)
    if (move.action == Move::ActionType::MOVE || move.action == Move::ActionType::PUSH) {
        int to_col = (move.action == Move::ActionType::MOVE) ? move.to.first : move.pushed_to.first;
        int to_row = (move.action == Move::ActionType::MOVE) ? move.to.second : move.pushed_to.second;
        if (to_col == 0 || to_col == 1 || to_col == cols - 2 || to_col == cols - 1) {
            priority += 300; 
            int target_row = (current_player == "circle") ? top_score_row() : bottom_score_row(rows);
            if (current_player == "circle" && (to_row > target_row && to_row < target_row + 3)) priority += 200;
            else if (current_player == "square" && (to_row < target_row && to_row > target_row - 3)) priority += 200;
        }
    }

    if (move.action == Move::ActionType::FLIP) {
        int from_col = move.from.first; int from_row = move.from.second; int target_row = (current_player == "circle") ? top_score_row() : bottom_score_row(rows);
        if (current_player == "circle" && (from_row > target_row && from_row < target_row + 3)) { priority += 250; if (move.orientation == "horizontal") priority += 150; }
        else if (current_player == "square" && (from_row < target_row && from_row > target_row - 3)) { priority += 250; if (move.orientation == "horizontal") priority += 150; }
        if (from_col == 0 || from_col == 1 || from_col == cols - 2 || from_col == cols - 1) priority += 180;
    }
    
    return priority;
}

std::vector<Move> StudentAgent::order_moves_with_edge_control(const Board& board, std::vector<Move> moves, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols) {
    std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        return get_move_priority(board, a, current_player, rows, cols, score_cols) > 
               get_move_priority(board, b, current_player, rows, cols, score_cols);
    });
    return moves;
}

// ===================================================================
// STRATEGY & HELPER FUNCTIONS
// ===================================================================

std::string StudentAgent::get_game_phase(const Board& board, int rows, int cols, const std::vector<int>& score_cols) const {
    // Dynamic endgame threshold is (Score Area Size - 1).
    // This is 3 for small (4-space SA), 4 for medium (5-space SA), and 5 for large (6-space SA).
    size_t required_win_count = score_cols.size();
    size_t endgame_threshold = (required_win_count > 0) ? required_win_count - 1 : 0; 
    
    int my_stones = count_stones_in_score_area(board, this->player, rows, cols, score_cols);
    int opp_stones = count_stones_in_score_area(board, this->opponent, rows, cols, score_cols);
    
    // Endgame: One stone away from winning for either player
    if (my_stones >= endgame_threshold || opp_stones >= endgame_threshold) {
        return "endgame";
    }
    
    // Midgame: At least one stone scored by either player
    if (my_stones + opp_stones >= 1) { 
        return "midgame";
    }
    
    return "opening";
}
int StudentAgent::count_stones_in_score_area(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const {
    int count = 0; int score_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    for (int x : score_cols) {
        if (in_bounds(x, score_row, rows, cols)) {
            auto p = board[score_row][x];
            if (p && p->owner == player && p->side == "stone") count++;
        }
    }
    return count;
}
bool StudentAgent::is_winning_move(const Board& board, const Move& move, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    if (move.action != Move::ActionType::MOVE && move.action != Move::ActionType::PUSH) return false;
    
    // Determine the required win count based on the Score Area size
    size_t required_win_count = score_cols.size(); 

    // Temporarily apply the move
    Board temp_board = deep_copy_board(board);
    uint64_t dummy_hash = 0;
    Board& board_ref = temp_board;
    apply_move_inplace(board_ref, move, rows, cols, dummy_hash);
    
    // Check if the resulting state meets the required win condition
    return count_stones_in_score_area(temp_board, player, rows, cols, score_cols) >= required_win_count;
}
std::optional<Move> StudentAgent::find_blocking_move(const Board& board, const std::vector<Move>& my_moves, const Move& opp_winning_move, int rows, int cols, const std::vector<int>& score_cols) {
    std::pair<int, int> target_pos = {-1, -1};
    if (opp_winning_move.action == Move::ActionType::MOVE) target_pos = opp_winning_move.to;
    else if (opp_winning_move.action == Move::ActionType::PUSH) target_pos = opp_winning_move.pushed_to;
    if (target_pos.first == -1) return std::nullopt;
    for (const auto& move : my_moves) {
        if (move.action == Move::ActionType::MOVE && move.to == target_pos) return move;
        if (move.action == Move::ActionType::PUSH && move.pushed_to == target_pos) return move;
    }
    return std::nullopt;
}

// ===================================================================
// MAKE/UNMAKE FUNCTIONS (Unchanged)
// ===================================================================

UndoInfo StudentAgent::apply_move_inplace(Board& board, const Move& move, int rows, int cols, uint64_t& hash) {
    int fr = move.from.second; int fc = move.from.first;
    auto piece = board[fr][fc];
    int piece_idx = get_zobrist_index(piece);
    int from_pos = fr * cols + fc;
    UndoInfo undo;
    undo.pushed_to = {-1, -1};
    board[fr][fc] = nullptr;
    hash ^= zobrist_table[piece_idx][from_pos];
    if (move.action == Move::ActionType::MOVE) {
        int tr = move.to.second; int tc = move.to.first;
        int to_pos = tr * cols + tc;
        undo.to = move.to;
        undo.piece_at_to = board[tr][tc];
        board[tr][tc] = piece;
        hash ^= zobrist_table[piece_idx][to_pos];
    } 
    else if (move.action == Move::ActionType::PUSH) {
        int tr = move.to.second; int tc = move.to.first;
        int ptr = move.pushed_to.second; int ptc = move.pushed_to.first;
        int to_pos = tr * cols + tc;
        int pushed_to_pos = ptr * cols + ptc;
        auto pushed_piece = board[tr][tc];
        int pushed_idx = get_zobrist_index(pushed_piece);
        undo.to = move.to;
        undo.pushed_to = move.pushed_to;
        undo.piece_at_to = pushed_piece;
        undo.piece_at_pushed_to = board[ptr][ptc];
        undo.original_side = piece->side;
        hash ^= zobrist_table[pushed_idx][to_pos];
        board[ptr][ptc] = pushed_piece;
        hash ^= zobrist_table[pushed_idx][pushed_to_pos];
        board[tr][tc] = piece;
        if (piece->side == "river") {
            piece->side = "stone";
            piece->orientation = "";
            int new_piece_idx = get_zobrist_index(piece);
            hash ^= zobrist_table[new_piece_idx][to_pos];
        } else {
            hash ^= zobrist_table[piece_idx][to_pos];
        }
    }
    else if (move.action == Move::ActionType::FLIP) {
        undo.original_side = piece->side;
        undo.original_orientation = piece->orientation;
        if (piece->side == "stone") {
            piece->side = "river";
            piece->orientation = move.orientation;
        } else {
            piece->side = "stone";
            piece->orientation = "";
        }
        int new_piece_idx = get_zobrist_index(piece);
        board[fr][fc] = piece;
        hash ^= zobrist_table[new_piece_idx][from_pos];
    }
    else if (move.action == Move::ActionType::ROTATE) {
        undo.original_orientation = piece->orientation;
        piece->orientation = (piece->orientation == "horizontal") ? "vertical" : "horizontal";
        int new_piece_idx = get_zobrist_index(piece);
        board[fr][fc] = piece;
        hash ^= zobrist_table[new_piece_idx][from_pos];
    }
    return undo;
}

void StudentAgent::unapply_move(Board& board, const Move& move, const UndoInfo& undo, int rows, int cols, uint64_t& hash) {
    int fr = move.from.second; int fc = move.from.first;
    if (move.action == Move::ActionType::MOVE) {
        int tr = move.to.second; int tc = move.to.first;
        auto piece = board[tr][tc];
        hash ^= zobrist_table[get_zobrist_index(piece)][tr * cols + tc];
        board[tr][tc] = undo.piece_at_to;
        hash ^= zobrist_table[get_zobrist_index(piece)][fr * cols + fc];
        board[fr][fc] = piece;
    }
    else if (move.action == Move::ActionType::PUSH) {
        int tr = move.to.second; int tc = move.to.first;
        int ptr = move.pushed_to.second; int ptc = move.pushed_to.first;
        auto moving_piece = board[tr][tc];
        auto pushed_piece = board[ptr][ptc];
        hash ^= zobrist_table[get_zobrist_index(moving_piece)][tr * cols + tc];
        board[tr][tc] = undo.piece_at_to;
        hash ^= zobrist_table[get_zobrist_index(pushed_piece)][tr * cols + tc];
        hash ^= zobrist_table[get_zobrist_index(pushed_piece)][ptr * cols + ptc];
        board[ptr][ptc] = undo.piece_at_pushed_to;
        moving_piece->side = undo.original_side;
        moving_piece->orientation = undo.original_orientation;
        hash ^= zobrist_table[get_zobrist_index(moving_piece)][fr * cols + fc];
        board[fr][fc] = moving_piece;
    }
    else if (move.action == Move::ActionType::FLIP || move.action == Move::ActionType::ROTATE) {
        auto piece = board[fr][fc];
        hash ^= zobrist_table[get_zobrist_index(piece)][fr * cols + fc];
        piece->side = undo.original_side;
        piece->orientation = undo.original_orientation;
        hash ^= zobrist_table[get_zobrist_index(piece)][fr * cols + fc];
        board[fr][fc] = piece;
    }
}


// ===================================================================
// MOVE GENERATION (Unchanged)
// ===================================================================

std::vector<Move> StudentAgent::get_all_valid_moves_enhanced(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const {
    std::vector<Move> moves;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (board[r][c] && board[r][c]->owner == player) {
                auto piece_moves = _get_moves_for_piece(board, r, c, player, rows, cols, score_cols);
                moves.insert(moves.end(), piece_moves.begin(), piece_moves.end());
            }
        }
    }
    return moves;
}

std::vector<Move> StudentAgent::_get_moves_for_piece(const Board& board, int row, int col, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const {
    std::vector<Move> moves; auto piece = board[row][col]; std::pair<int,int> from_pos = {col, row};
    int dr[] = {-1, 1, 0, 0}; int dc[] = {0, 0, -1, 1};
    for (int i = 0; i < 4; ++i) {
        int nr = row + dr[i]; int nc = col + dc[i];
        if (!in_bounds(nc, nr, rows, cols) || is_opponent_score_cell(nc, nr, player, rows, cols, score_cols)) continue;
        auto target_cell = board[nr][nc];
        if (!target_cell) moves.emplace_back(Move::ActionType::MOVE, from_pos, std::make_pair(nc, nr));
        else if (target_cell->side == "river") {
            auto flow_dests = _trace_river_flow(board, nr, nc, player, rows, cols, score_cols);
            for (auto dest : flow_dests) moves.emplace_back(Move::ActionType::MOVE, from_pos, std::make_pair(dest.second, dest.first));
        } else if (target_cell->side == "stone") {
            if (piece->side == "stone") {
                int pr = nr + dr[i]; int pc = nc + dc[i];
                if (in_bounds(pc, pr, rows, cols) && !board[pr][pc] && !is_opponent_score_cell(pc, pr, target_cell->owner, rows, cols, score_cols)) {
                    moves.emplace_back(Move::ActionType::PUSH, from_pos, std::make_pair(nc, nr), std::make_pair(pc, pr));
                }
            } else {
                auto push_dests = _trace_river_push(board, nr, nc, piece, target_cell->owner, rows, cols, score_cols);
                for (auto dest : push_dests) moves.emplace_back(Move::ActionType::PUSH, from_pos, std::make_pair(nc, nr), std::make_pair(dest.second, dest.first));
            }
        }
    }
    if (piece->side == "stone") {
        moves.emplace_back(Move::ActionType::FLIP, from_pos, "horizontal");
        moves.emplace_back(Move::ActionType::FLIP, from_pos, "vertical");
    } else {
        moves.emplace_back(Move::ActionType::FLIP, from_pos);
        moves.emplace_back(Move::ActionType::ROTATE, from_pos);
    }
    return moves;
}

std::vector<std::pair<int, int>> StudentAgent::_trace_river_flow(const Board& board, int start_r, int start_c, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const {
    std::deque<std::pair<int, int>> q = {{start_r, start_c}}; std::set<std::pair<int, int>> visited_rivers = {{start_r, start_c}}; std::set<std::pair<int, int>> destinations;
    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop_front(); auto river = board[r][c]; if (!river || river->side != "river") continue;
        std::vector<std::pair<int, int>> dirs; if (river->orientation == "vertical") dirs = {{-1, 0}, {1, 0}}; else dirs = {{0, -1}, {0, 1}};
        for (auto [dr, dc] : dirs) {
            int cr = r + dr; int cc = c + dc;
            while (in_bounds(cc, cr, rows, cols)) {
                if (is_opponent_score_cell(cc, cr, player, rows, cols, score_cols)) break;
                auto cell = board[cr][cc];
                if (!cell) destinations.insert({cr, cc});
                else {
                    if (cell->side == "river" && visited_rivers.find({cr, cc}) == visited_rivers.end()) { visited_rivers.insert({cr, cc}); q.push_back({cr, cc}); }
                    break;
                }
                cr += dr; cc += dc;
            }
        }
    }
    return std::vector<std::pair<int, int>>(destinations.begin(), destinations.end());
}

std::vector<std::pair<int, int>> StudentAgent::_trace_river_push(const Board& board, int target_r, int target_c, const PiecePtr& river_piece, const std::string& pushed_player, int rows, int cols, const std::vector<int>& score_cols) const {
    std::set<std::pair<int, int>> destinations; std::vector<std::pair<int, int>> dirs;
    if (river_piece->orientation == "vertical") dirs = {{-1, 0}, {1, 0}}; else dirs = {{0, -1}, {0, 1}};
    for (auto [dr, dc] : dirs) {
        int cr = target_r + dr; int cc = target_c + dc;
        while (in_bounds(cc, cr, rows, cols)) {
            if (is_opponent_score_cell(cc, cr, pushed_player, rows, cols, score_cols)) break;
            auto cell = board[cr][cc];
            if (!cell) destinations.insert({cr, cc});
            else if (cell->side != "river") break;
            cr += dr; cc += dc;
        }
    }
    return std::vector<std::pair<int, int>>(destinations.begin(), destinations.end());
}


// ===================================================================
// PYBIND11 MODULE BINDINGS (Unchanged)
// ===================================================================

PYBIND11_MODULE(student_agent_module, m) {
    m.doc() = "pybind11 module for StudentAgent";

    py::class_<Piece, std::shared_ptr<Piece>>(m, "Piece")
        .def(py::init<std::string, std::string, std::string>(),
             py::arg("owner"), py::arg("side"), py::arg("orientation") = "")
        .def_readwrite("owner", &Piece::owner)
        .def_readwrite("side", &Piece::side)
        .def_readwrite("orientation", &Piece::orientation);

    py::enum_<Move::ActionType>(m, "ActionType")
        .value("MOVE", Move::ActionType::MOVE)
        .value("PUSH", Move::ActionType::PUSH)
        .value("FLIP", Move::ActionType::FLIP)
        .value("ROTATE", Move::ActionType::ROTATE)
        .export_values();

    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("action", &Move::action)
        .def_readwrite("from_pos", &Move::from) 
        .def_readwrite("to_pos", &Move::to)
        .def_readwrite("pushed_to", &Move::pushed_to)
        .def_readwrite("orientation", &Move::orientation);

    py::class_<StudentAgent>(m, "StudentAgent") 
        .def(py::init<std::string>())
        .def("choose", &StudentAgent::choose,
             py::arg("board"), py::arg("rows"), py::arg("cols"), 
             py::arg("score_cols"), py::arg("current_player_time"), 
             py::arg("opponent_time"));
}


