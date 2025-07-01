/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {
// Custom Point Chess piece values
constexpr int PointChessValues[PIECE_TYPE_NB] = {0, 100, 300, 300, 500, 900, 0}; // None, Pawn, Knight, Bishop, Rook, Queen, King
// Returns a static, purely materialistic evaluation of the position from
// the point of view of the side to move. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position& pos) {
    int material = 0;
    for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
        material += PointChessValues[pt] * (pos.count(pt, WHITE) - pos.count(pt, BLACK));
    }
    return material;
}

bool Eval::use_smallnet(const Position& pos) { return std::abs(simple_eval(pos)) > 962; }

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks& networks,
                     const Position& pos,
                     Eval::NNUE::AccumulatorStack& accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int optimism)
{
    assert(!pos.checkers());

    // First check for checkmate
    if (pos.is_checkmate())
        return pos.side_to_move() == WHITE ? VALUE_MATE : -VALUE_MATE;

    // Calculate material difference using Point Chess values
    int material = 0;
    for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
        material += PointChessValues[pt] * (pos.count(pt, WHITE) - pos.count(pt, BLACK));
    }

    // Reward captures and safe exchanges
    int captureBonus = 0;
    for (Move m : pos.captures()) {
        PieceType captured = type_of(pos.piece_on(to_sq(m)));
        PieceType attacker = type_of(pos.moved_piece(m));
        if (PointChessValues[attacker] <= PointChessValues[captured]) {
            captureBonus += PointChessValues[captured] * 2; // Double value for good captures
        }
    }

    // Get original NNUE evaluation (keep some positional awareness)
    bool smallNet = use_smallnet(pos);
    auto [psqt, positional] = smallNet ? networks.small.evaluate(pos, accumulators, caches.small)
                                       : networks.big.evaluate(pos, accumulators, caches.big);
    Value nnue = (125 * psqt + 131 * positional) / 128;

    // Combine evaluations with heavy weight on material
    int v = (material * 1000) + (captureBonus * 500) + nnue;

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Networks& networks) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    Eval::NNUE::AccumulatorStack accumulators;
    auto                         caches = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, networks, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    auto [psqt, positional] = networks.big.evaluate(pos, accumulators, &caches->big);
    Value v                 = psqt + positional;
    v                       = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = evaluate(networks, pos, accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "Final evaluation       " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]";
    ss << "\n";

    return ss.str();
}

}  // namespace Stockfish
