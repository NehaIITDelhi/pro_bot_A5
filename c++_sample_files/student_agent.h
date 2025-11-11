#ifndef STUDENT_AGENT_H
#define STUDENT_AGENT_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <deque>
#include <algorithm>
#include <chrono> // For time
#include <functional> // For std::hash
#include <limits> // For infinity
#include <random> // For Zobrist Hashing

// ===================================================================
// HELPER STRUCTURES (Unchanged)
// ===================================================================
struct Piece {
    std::string owner;
    std::string side;
    std::string orientation;
    Piece(std::string o, std::string s, std::string orient = "")
        : owner(std::move(o)), side(std::move(s)), orientation(std::move(orient)) {}
};
using PiecePtr = std::shared_ptr<Piece>;
using Board = std::vector<std::vector<PiecePtr>>;
struct Move {
    enum class ActionType { MOVE, PUSH, FLIP, ROTATE };
    ActionType action;
    std::pair<int, int> from;
    std::pair<int, int> to;
    std::pair<int, int> pushed_to;
    std::string orientation;
    Move() : action(ActionType::MOVE), from({0,0}), to({0,0}), pushed_to({0,0}), orientation("") {}
    Move(ActionType a, std::pair<int, int> f, std::pair<int, int> t) : action(a), from(f), to(t), pushed_to({-1,-1}), orientation("") {}
    Move(ActionType a, std::pair<int, int> f, std::pair<int, int> t, std::pair<int, int> pt) : action(a), from(f), to(t), pushed_to(pt), orientation("") {}
    Move(ActionType a, std::pair<int, int> f, std::string o) : action(a), from(f), to({-1,-1}), pushed_to({-1,-1}), orientation(o) {}
    Move(ActionType a, std::pair<int, int> f) : action(a), from(f), to({-1,-1}), pushed_to({-1,-1}), orientation("") {}
    bool operator==(const Move& other) const {
        return action == other.action && from == other.from && to == other.to && pushed_to == other.pushed_to && orientation == other.orientation;
    }
};
struct UndoInfo {
    std::pair<int, int> to;
    std::pair<int, int> pushed_to;
    PiecePtr piece_at_to;
    PiecePtr piece_at_pushed_to;
    std::string original_side;
    std::string original_orientation;
};
enum class TTFlag { EXACT, LOWER_BOUND, UPPER_BOUND };
struct TTEntry {
    double score;
    int depth;
    TTFlag flag;
};

// ===================================================================
// BASE AGENT (Unchanged)
// ===================================================================
class BaseAgent {
public:
    std::string player;
    std::string opponent;
    BaseAgent(std::string p) : player(p), opponent(get_opponent(p)) {}
    virtual ~BaseAgent() {}
    virtual std::optional<Move> choose(
        const Board& board, int rows, int cols, const std::vector<int>& score_cols,
        double current_player_time, double opponent_time) = 0;
    static std::string get_opponent(const std::string& p) {
        return (p == "circle") ? "square" : "circle";
    }
};

// ===================================================================
// STUDENT AGENT CLASS
// ===================================================================
class StudentAgent : public BaseAgent {
private:
    std::map<int, std::vector<Move>> killer_moves;
    
    // --- MODIFIED: Storing Zobrist hashes ---
    static std::deque<uint64_t> recent_positions;
    static const int MAX_HISTORY_SIZE = 20;
    // --- END MODIFIED ---

    std::deque<Move> last_moves;
    const int MAX_RECENT_MOVES = 5;
    
    int turn_count;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
    double time_limit_seconds;

    // --- Zobrist & Transposition Table ---
    static const int MAX_BOARD_SIZE = 20 * 20;
    static const int NUM_PIECE_TYPES = 6;
    std::map<uint64_t, TTEntry> transposition_table;
    std::vector<std::vector<uint64_t>> zobrist_table;
    std::mt19937_64 random_engine;
    
    void init_zobrist(int rows, int cols);
    int get_zobrist_index(const PiecePtr& piece) const;
    uint64_t board_hash_zobrist(const Board& board, int rows, int cols) const;

    // --- Utility Functions ---
    bool in_bounds(int x, int y, int rows, int cols) const;
    std::vector<int> score_cols_for(int cols) const;
    int top_score_row() const;
    int bottom_score_row(int rows) const;
    bool is_opponent_score_cell(int x, int y, const std::string& p, int rows, int cols, const std::vector<int>& score_cols) const;
    bool is_own_score_cell(int x, int y, const std::string& p, int rows, int cols, const std::vector<int>& score_cols) const;
    // size_t board_hash(const Board& board) const; // <-- REMOVED slow hash
    int manhattan_distance(int x1, int y1, int x2, int y2) const;
    Board deep_copy_board(const Board& board);

    // --- Main Logic ---
    void update_move_history(const Move& move);
    bool moves_similar(const Move& move1, const Move& move2) const;
    
    double negamax_with_balance(Board& board, int depth, double alpha, double beta,
                                const std::string& current_player, int rows, int cols, 
                                const std::vector<int>& score_cols, int max_depth, 
                                uint64_t& zobrist_hash);
    
    double quiescence_search(Board& board, double alpha, double beta,
                             const std::string& current_player, int rows, int cols,
                             const std::vector<int>& score_cols, uint64_t& zobrist_hash, int q_depth);

    // --- Evaluation ---
    double evaluate_balanced(const Board& board, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols);
    double evaluate_edge_control(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    double evaluate_defensive_position(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    double evaluate_balance_factor(const Board& board, const std::string& player, int rows, int cols);
    double evaluate_river_network_balanced(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    double evaluate_manhattan_distances(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    double count_stones_ready_to_score(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);

    // --- Move Generation & Ordering ---
    int get_move_priority(const Board& board, const Move& move, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols);
    std::vector<Move> order_moves_with_edge_control(const Board& board, std::vector<Move> moves, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols);
    std::vector<Move> get_all_valid_moves_enhanced(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const;
    std::vector<Move> _get_moves_for_piece(const Board& board, int row, int col, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const;
    std::vector<std::pair<int, int>> _trace_river_flow(const Board& board, int start_r, int start_c, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const;
    std::vector<std::pair<int, int>> _trace_river_push(const Board& board, int target_r, int target_c, const PiecePtr& river_piece, const std::string& pushed_player, int rows, int cols, const std::vector<int>& score_cols) const;

    // --- Strategy & Helpers ---
    std::string get_game_phase(const Board& board, int rows, int cols, const std::vector<int>& score_cols) const;
    int count_stones_in_score_area(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const;
    bool is_winning_move(const Board& board, const Move& move, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    std::optional<Move> find_blocking_move(const Board& board, const std::vector<Move>& my_moves, const Move& opp_winning_move, int rows, int cols, const std::vector<int>& score_cols);
    
    // --- Make/Unmake move functions ---
    UndoInfo apply_move_inplace(Board& board, const Move& move, int rows, int cols, uint64_t& hash);
    void unapply_move(Board& board, const Move& move, const UndoInfo& undo, int rows, int cols, uint64_t& hash);

public:
    StudentAgent(std::string p);

    std::optional<Move> choose(
        const Board& board, int rows, int cols, const std::vector<int>& score_cols,
        double current_player_time, double opponent_time) override;
};

#endif // STUDENT_AGENT_H
