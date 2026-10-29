import streamlit as st
import streamlit.components.v1 as components
import chess
import chess.pgn
import io
import joblib
import pandas as pd
import numpy as np
from stockfish import Stockfish
from st_bridge import bridge
from modules.chess import Chess
import json
import plotly.graph_objects as go
import base64
import chess.svg

# ---------- UI helper: W/D/B progress bars ----------
def render_probability_bars(p_white, p_draw, p_black):
    rows = [
        ("⚪ White", p_white, "#22c55e"),   # bright green
        ("➖ Draw",  p_draw,  "#a855f7"),   # purple
        ("⚫ Black", p_black, "#3b82f6"),   # blue
    ]
    for label, prob, color in rows:
        c1, c2 = st.columns([3, 1])
        c1.markdown(
            f"<div style='color:#c9d1d9;font-size:0.9rem;"
            f"margin-bottom:2px;font-weight:600;'>{label}</div>",
            unsafe_allow_html=True)
        c2.markdown(
            f"<div style='color:#e6edf3;font-size:0.95rem;text-align:right;"
            f"font-weight:700;'>{prob:.0%}</div>",
            unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#1a2332;border-radius:8px;height:12px;"
            f"margin-top:-6px;margin-bottom:12px;overflow:hidden;'>"
            f"<div style='background:{color};width:{prob*100:.1f}%;"
            f"height:100%;border-radius:8px;"
            f"box-shadow: 0 0 10px {color}66;'></div></div>",
            unsafe_allow_html=True)
def render_gauge(value, label, color="#22c55e"):
    """Pure SVG donut gauge — no Plotly, no zoom/disappear bugs."""
    pct = value * 100
    radius = 70
    circumference = 2 * 3.14159 * radius
    dash = (pct / 100) * circumference
    svg = f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;padding:14px 0;">
      <svg width="220" height="220" viewBox="0 0 200 200">
        <!-- Track ring -->
        <circle cx="100" cy="100" r="{radius}"
                stroke="#1a2332" stroke-width="22" fill="none" />
        <!-- Value ring -->
        <circle cx="100" cy="100" r="{radius}"
                stroke="{color}" stroke-width="22" fill="none"
                stroke-dasharray="{dash} {circumference}"
                stroke-dashoffset="0"
                transform="rotate(-90 100 100)"
                stroke-linecap="round"
                style="filter: drop-shadow(0 0 10px {color}aa);" />
        <!-- Center % -->
        <text x="100" y="100" text-anchor="middle"
              dominant-baseline="central"
              fill="{color}"
              font-size="42"
              font-weight="900"
              font-family="Arial Black, Helvetica, sans-serif">
          {pct:.0f}%
        </text>
      </svg>
      <div style="color:#e6edf3;font-size:1.1rem;font-weight:700;
                  margin-top:6px;">{label}</div>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


import os
import sys

def get_app_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------- Configuration ----------
STOCKFISH_PATH = get_app_path(os.path.join("stockfish", "stockfish-windows-x86-64-universal.exe"))
MODEL_PATH = get_app_path("chess_model.pkl")
FEATURES_PATH = get_app_path("feature_cols.pkl")
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# ---------- Load assets ----------
@st.cache_resource
def load_assets():
    clf = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    # Added get_app_path() to these three!
    prematch_model = joblib.load(get_app_path("prematch_model.pkl"))
    prematch_model_unb = joblib.load(get_app_path("prematch_model_unbalanced.pkl"))
    prematch_features = joblib.load(get_app_path("prematch_feature_cols.pkl"))
    
    sf = Stockfish(path=STOCKFISH_PATH, depth=10, parameters={"Threads": 2, "Hash": 128})
    return clf, feature_cols, sf, prematch_model, prematch_model_unb, prematch_features

clf, feature_cols, sf, prematch_model, prematch_model_unb, prematch_features = load_assets()


# ---------- Feature helpers ----------
def board_material(board):
    mat_w = mat_b = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        if p.color == chess.WHITE:
            mat_w += PIECE_VALUES[p.piece_type]
        else:
            mat_b += PIECE_VALUES[p.piece_type]
    return mat_w, mat_b


def stockfish_best_move(board, depth=15):
    sf.set_fen_position(board.fen())
    best_uci = sf.get_best_move()
    if not best_uci:
        return None
    try:
        move = chess.Move.from_uci(best_uci)
        san = board.san(move)
        return {"uci": best_uci, "san": san}
    except Exception:
        return {"uci": best_uci, "san": best_uci}


def stockfish_eval_white_pov(board):
    sf.set_fen_position(board.fen())
    ev = sf.get_evaluation()
    if ev["type"] == "cp":
        cp = ev["value"]
    elif ev["type"] == "mate":
        cp = 10000 if ev["value"] > 0 else -10000
    else:
        cp = 0
    if board.turn == chess.BLACK:
        cp = -cp
    return cp


def _eval_cp_white_pov(board):
    sf.set_fen_position(board.fen())
    ev = sf.get_evaluation()
    if ev["type"] == "cp":
        cp = ev["value"]
    elif ev["type"] == "mate":
        cp = 10000 if ev["value"] > 0 else -10000
    else:
        cp = 0
    if board.turn == chess.BLACK:
        cp = -cp
    return cp


def build_features(board, white_elo, black_elo, ply):
    mat_w, mat_b = board_material(board)
    cp = stockfish_eval_white_pov(board)
    return {
        "material_balance": mat_w - mat_b,
        "white_bishops": len(board.pieces(chess.BISHOP, chess.WHITE)),
        "black_bishops": len(board.pieces(chess.BISHOP, chess.BLACK)),
        "total_pieces": len(board.piece_map()),
        "ply": ply,
        "elo_diff": white_elo - black_elo,
        "stockfish_eval": cp,
        "stockfish_vs_material": cp - (mat_w - mat_b) * 100,
    }


def predict_from_features(feats):
    cp = feats["stockfish_eval"]
    if cp == 10000:
        return {"label": "White wins", "confidence": 0.97,
                "p_white": 0.97, "p_draw": 0.02, "p_black": 0.01, "cp": cp}
    if cp == -10000:
        return {"label": "Black wins", "confidence": 0.97,
                "p_white": 0.01, "p_draw": 0.02, "p_black": 0.97, "cp": cp}
    if cp >= 500:
        return {"label": "White wins", "confidence": 0.90,
                "p_white": 0.90, "p_draw": 0.08, "p_black": 0.02, "cp": cp}
    if cp <= -500:
        return {"label": "Black wins", "confidence": 0.90,
                "p_white": 0.02, "p_draw": 0.08, "p_black": 0.90, "cp": cp}
    X_new = pd.DataFrame([feats])[feature_cols]
    proba = clf.predict_proba(X_new)[0]
    pred_class = clf.classes_[np.argmax(proba)]
    probs = dict(zip(clf.classes_, proba))
    label = {1: "White wins", 0: "Draw", -1: "Black wins"}[pred_class]
    return {
        "label": label, "confidence": max(proba),
        "p_white": probs.get(1, 0), "p_draw": probs.get(0, 0),
        "p_black": probs.get(-1, 0), "cp": cp,
    }


def predict_from_pgn(pgn_string, snapshot_ply=80):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return {"error": "Could not parse PGN"}
    headers = game.headers
    white_elo = int(headers.get("WhiteElo", 0) or 0)
    black_elo = int(headers.get("BlackElo", 0) or 0)
    board = game.board()
    moves = list(game.mainline_moves())
    for i, mv in enumerate(moves):
        if i >= snapshot_ply:
            break
        board.push(mv)
    ply = min(snapshot_ply, len(moves))
    feats = build_features(board, white_elo, black_elo, ply)
    pred = predict_from_features(feats)
    return {
        "white": headers.get("White", "?"), "black": headers.get("Black", "?"),
        "white_elo": white_elo, "black_elo": black_elo,
        "snapshot_ply": ply, "material_balance": feats["material_balance"],
        "stockfish_eval": feats["stockfish_eval"], "fen": board.fen(), **pred,
    }


def predict_from_fen(fen_string, white_elo, black_elo):
    try:
        board = chess.Board(fen_string.strip())
    except ValueError as e:
        return {"error": f"Invalid FEN: {e}"}
    actual_ply = (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
    if actual_ply < 60 or actual_ply > 100:
        ply = 80
        ply_warning = f"⚠️ Position is at ply {actual_ply}. Model was trained on mid-game (~ply 80). Results may be less reliable."
    else:
        ply = actual_ply
        ply_warning = None
    feats = build_features(board, white_elo, black_elo, ply)
    pred = predict_from_features(feats)
    return {
        "white_elo": white_elo, "black_elo": black_elo,
        "snapshot_ply": ply, "actual_ply": actual_ply,
        "ply_warning": ply_warning,
        "material_balance": feats["material_balance"],
        "stockfish_eval": feats["stockfish_eval"],
        "fen": board.fen(), **pred,
    }


def _raw_model_predict(model, elo_diff, avg_elo):
    X = pd.DataFrame([{
        "elo_diff": elo_diff, "avg_elo": avg_elo, "white_advantage": 1,
    }])[prematch_features]
    proba = model.predict_proba(X)[0]
    probs = dict(zip(model.classes_, proba))
    pred = model.classes_[np.argmax(proba)]
    label = {1: "White wins", 0: "Draw", -1: "Black wins"}[pred]
    return {
        "label": label, "confidence": max(proba),
        "p_white": probs.get(1, 0), "p_draw": probs.get(0, 0),
        "p_black": probs.get(-1, 0),
    }


def _add_decisive(p):
    total = p["p_white"] + p["p_black"]
    if total > 0.01:
        p["w_if_decisive"] = p["p_white"] / total
        p["b_if_decisive"] = p["p_black"] / total
    else:
        p["w_if_decisive"] = None
        p["b_if_decisive"] = None
    return p


def predict_prematch_both(white_elo, black_elo):
    elo_diff = white_elo - black_elo
    avg_elo = (white_elo + black_elo) / 2
    if abs(elo_diff) < 30:
        shared = {"label": "Draw", "confidence": 0.55,
                  "p_white": 0.28, "p_draw": 0.55, "p_black": 0.17,
                  "source": "heuristic (even matchup)"}
        return {"unbalanced": _add_decisive(dict(shared)),
                "balanced": _add_decisive(dict(shared))}
    if abs(elo_diff) >= 400:
        if elo_diff > 0:
            shared = {"label": "White wins", "confidence": 0.92,
                      "p_white": 0.92, "p_draw": 0.05, "p_black": 0.03,
                      "source": "override (White much stronger)"}
        else:
            shared = {"label": "Black wins", "confidence": 0.92,
                      "p_white": 0.03, "p_draw": 0.05, "p_black": 0.92,
                      "source": "override (Black much stronger)"}
        return {"unbalanced": _add_decisive(dict(shared)),
                "balanced": _add_decisive(dict(shared))}
    unb = _raw_model_predict(prematch_model_unb, elo_diff, avg_elo)
    unb["source"] = "unbalanced ML"
    bal = _raw_model_predict(prematch_model, elo_diff, avg_elo)
    bal["source"] = "balanced ML"
    return {"unbalanced": _add_decisive(unb), "balanced": _add_decisive(bal)}


def eval_to_win_prob(cp):
    return 1.0 / (1.0 + np.exp(-cp / 400.0))


def _rebuild_pgn(moves_list):
    parts = []
    for i in range(0, len(moves_list), 2):
        num = i // 2 + 1
        w = moves_list[i][1] if i < len(moves_list) and moves_list[i][0] == "w" else ""
        b = moves_list[i+1][1] if i+1 < len(moves_list) and moves_list[i+1][0] == "b" else ""
        parts.append(f"{num}. {w} {b}".strip())
    return " ".join(parts)


def explain_move(move_info, fen_before):
    san = move_info.get("san", "?")
    color = move_info.get("color", "?")
    cat = move_info.get("category", "?")
    cpl = move_info.get("cpl", 0)
    best = move_info.get("best_alt", "")
    try:
        board = chess.Board(fen_before)
        mv = board.parse_san(san)
        piece = board.piece_at(mv.from_square)
        captured = board.piece_at(mv.to_square)
        is_check = board.gives_check(mv) if hasattr(board, 'gives_check') else False
        is_castle = board.is_castling(mv)
    except Exception:
        piece = None; captured = None; is_check = False; is_castle = False
    piece_names = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
                   chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
    what = ""
    if is_castle:
        what = "Castles the king to safety."
    elif captured:
        what = f"Captures the {piece_names.get(captured.piece_type, 'piece')}."
    elif is_check:
        what = "Gives check."
    elif piece:
        p = piece_names.get(piece.piece_type, "piece")
        what = f"A strong {p} move." if cat in ("Best", "Excellent") else f"Moves the {p}."
    if cat == "Best":
        why = "This is the engine's top choice — no better move exists."
    elif cat == "Excellent":
        why = "Nearly the best move available."
    elif cat == "Good":
        why = "A solid move."
    elif cat == "Inaccuracy":
        why = f"Slightly imprecise — lost {cpl/100:.1f} pawns of advantage."
    elif cat == "Mistake":
        why = f"This is a mistake — it drops {cpl/100:.1f} pawns."
    elif cat == "Blunder":
        why = f"This is a blunder — it gives away {cpl/100:.1f} pawns."
    else:
        why = ""
    alt = ""
    if best and cat in ("Inaccuracy", "Mistake", "Blunder"):
        alt = f"Better was {best}."
    parts = [p for p in [what, why, alt] if p]
    return " ".join(parts) if parts else "No explanation available."


# ============================================================
# OPENING DETECTION
# ============================================================
_ECO_CACHE = None

def load_eco_database():
    global _ECO_CACHE
    if _ECO_CACHE is not None:
        return _ECO_CACHE
    import os
    entries = []
    for letter in ["a", "b", "c", "d", "e"]:
        path = get_app_path(f"eco_{letter}.tsv")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    eco = parts[0].strip()
                    name = parts[1].strip()
                    moves_str = parts[2].strip()
                    normalized = normalize_moves(moves_str)
                    if normalized:
                        entries.append((normalized, name, eco))
        except Exception:
            continue
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    _ECO_CACHE = entries
    return entries


def normalize_moves(moves_str):
    import re
    cleaned = re.sub(r"\d+\.+", " ", moves_str)
    tokens = cleaned.split()
    moves = []
    for t in tokens:
        t = t.strip()
        if not t or t in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        moves.append(t)
    return moves


def detect_opening(pgn_string):
    entries = load_eco_database()
    if not entries:
        return None, None
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return None, None
    board = game.board()
    game_moves = []
    for mv in game.mainline_moves():
        try:
            san = board.san(mv)
            game_moves.append(san)
            board.push(mv)
        except Exception:
            break
        if len(game_moves) >= 20:
            break
    if not game_moves:
        return None, None
    for moves_tuple, name, eco in entries:
        n = len(moves_tuple)
        if n <= len(game_moves):
            if game_moves[:n] == moves_tuple:
                return name, eco
    return None, None


def classify_cpl(cpl, eval_before=0, eval_after=0, mover_is_white=True):
    if mover_is_white:
        e_before = eval_before
        e_after = eval_after
    else:
        e_before = -eval_before
        e_after = -eval_after
    if e_before >= 300 and e_after >= 200:
        if cpl <= 5: return "⭐ Best", "Best"
        elif cpl <= 20: return "🌟 Excellent", "Excellent"
        elif cpl <= 50: return "✅ Good", "Good"
        else: return "⚠️ Inaccuracy", "Inaccuracy"
    if e_before <= -300 and e_after <= -200:
        if cpl <= 5: return "⭐ Best", "Best"
        elif cpl <= 20: return "🌟 Excellent", "Excellent"
        elif cpl <= 50: return "✅ Good", "Good"
        else: return "⚠️ Inaccuracy", "Inaccuracy"
    if cpl <= 5: return "⭐ Best", "Best"
    elif cpl <= 20: return "🌟 Excellent", "Excellent"
    elif cpl <= 50: return "✅ Good", "Good"
    elif cpl <= 100: return "⚠️ Inaccuracy", "Inaccuracy"
    elif cpl <= 300: return "❌ Mistake", "Mistake"
    else: return "💥 Blunder", "Blunder"


# ---- Heatmap / Chart Functions ----
def piece_activity_heatmap(fen):
    try:
        board = chess.Board(fen)
    except Exception:
        return [[0]*8 for _ in range(8)]
    grid = [[0]*8 for _ in range(8)]
    for sq in chess.SQUARES:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        try:
            white_att = len(board.attackers(chess.WHITE, sq))
            black_att = len(board.attackers(chess.BLACK, sq))
        except Exception:
            white_att = 0; black_att = 0
        grid[7 - rank][file] = white_att - black_att
    return grid


def piece_square_heatmap(pgn_string, piece_type=None):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return None, None
    white_grid = [[0]*8 for _ in range(8)]
    black_grid = [[0]*8 for _ in range(8)]
    board = game.board()
    for mv in game.mainline_moves():
        piece = board.piece_at(mv.from_square)
        if piece is not None:
            if piece_type is None or piece.piece_type == piece_type:
                for sq in (mv.from_square, mv.to_square):
                    f = chess.square_file(sq)
                    r = chess.square_rank(sq)
                    row = 7 - r
                    if piece.color == chess.WHITE:
                        white_grid[row][f] += 1
                    else:
                        black_grid[row][f] += 1
        board.push(mv)
    return white_grid, black_grid


def king_safety_score(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    score = 0
    file = chess.square_file(king_sq)
    rank = chess.square_rank(king_sq)
    pawn_dir = 1 if color == chess.WHITE else -1
    for df in (-1, 0, 1):
        f = file + df
        if f < 0 or f > 7: continue
        for dr in (1, 2):
            r = rank + pawn_dir * dr
            if r < 0 or r > 7: continue
            sq = chess.square(f, r)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                score += 3 if dr == 1 else 1
                break
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0: continue
            f = file + df; r = rank + dr
            if f < 0 or f > 7 or r < 0 or r > 7: continue
            sq = chess.square(f, r)
            score += len(board.attackers(color, sq)) * 0.3
    enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
    enemy_attacks = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f = file + df; r = rank + dr
            if f < 0 or f > 7 or r < 0 or r > 7: continue
            sq = chess.square(f, r)
            enemy_attacks += len(board.attackers(enemy, sq))
    score -= enemy_attacks * 1.5
    return max(0, score)


def king_safety_timeline(fens):
    white_scores = []
    black_scores = []
    for fen in fens:
        try:
            b = chess.Board(fen)
            white_scores.append(round(king_safety_score(b, chess.WHITE), 1))
            black_scores.append(round(king_safety_score(b, chess.BLACK), 1))
        except Exception:
            white_scores.append(0); black_scores.append(0)
    return white_scores, black_scores


def material_balance_from_fen(fen):
    try:
        board = chess.Board(fen)
    except Exception:
        return 0
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    diff = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None: continue
        v = values.get(p.piece_type, 0)
        diff += v if p.color == chess.WHITE else -v
    return diff


def compute_eval_graph(pgn_string, max_plies=120, depth=10):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return {"error": "Could not parse PGN"}
    try:
        sf.set_depth(depth)
    except Exception:
        pass
    board = game.board()
    moves = list(game.mainline_moves())[:max_plies]
    evals = [_eval_cp_white_pov(board)]
    fens = [board.fen()]
    sans = ["(start)"]
    cpls = [0]
    categories = ["Start"]
    best_alts = [""]
    for mv in moves:
        mover_is_white = board.turn == chess.WHITE
        san = board.san(mv)
        fen_before = board.fen()
        before_cp = evals[-1]
        sf.set_fen_position(fen_before)
        best_uci = sf.get_best_move() or ""
        best_san = ""
        if best_uci:
            try:
                best_san = board.san(chess.Move.from_uci(best_uci))
            except Exception:
                best_san = ""
        board.push(mv)
        after_cp = _eval_cp_white_pov(board)
        if mover_is_white:
            cpl = max(0, before_cp - after_cp)
        else:
            cpl = max(0, after_cp - before_cp)
        if best_uci and mv.uci() == best_uci:
            cpl = 0
        label, category = classify_cpl(cpl, before_cp, after_cp, mover_is_white)
        evals.append(after_cp)
        fens.append(board.fen())
        sans.append(san)
        cpls.append(cpl)
        categories.append(category)
        best_alts.append(best_san)
    try:
        sf.set_depth(15)
    except Exception:
        pass
    return {
        "evals": evals, "fens": fens, "sans": sans,
        "cpls": cpls, "categories": categories,
        "best_alts": best_alts, "num_moves": len(moves),
    }


def analyze_game_moves(pgn_string, max_plies=120, depth=10):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return {"error": "Could not parse PGN"}
    try:
        sf.set_depth(depth)
    except Exception:
        pass
    board = game.board()
    moves = list(game.mainline_moves())[:max_plies]
    before_cp = _eval_cp_white_pov(board)
    results = []
    for i, mv in enumerate(moves):
        mover_is_white = board.turn == chess.WHITE
        san = board.san(mv)
        sf.set_fen_position(board.fen())
        best_uci = sf.get_best_move()
        board.push(mv)
        after_cp = _eval_cp_white_pov(board)
        if mover_is_white:
            cpl = max(0, before_cp - after_cp)
        else:
            cpl = max(0, after_cp - before_cp)
        if best_uci and mv.uci() == best_uci:
            cpl = 0
        label, category = classify_cpl(cpl, before_cp, after_cp, mover_is_white)
        results.append({
    "move_number": i // 2 + 1, "ply": i + 1,
    "color": "White" if mover_is_white else "Black",
    "san": san, "cpl": cpl, "label": label, "category": category,
    "eval_after": after_cp,
})
        before_cp = after_cp
    try:
        sf.set_depth(15)
    except Exception:
        pass
    return {"moves": results}


def render_analysis(r):
    if "error" in r:
        st.error(r["error"])
        return
    if r.get("ply_warning"):
        st.warning(r["ply_warning"])

    # ---------- Card 1: Position analysis ----------
    with st.container(border=True):
        st.markdown("### 📈 Position analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Material balance", f"{r['material_balance']:+d}")

        eval_cp = r["stockfish_eval"]
        if eval_cp >= 200:
            eval_label = f"🟢 +{eval_cp} cp (White winning)"
        elif eval_cp <= -200:
            eval_label = f"🔴 {eval_cp} cp (Black winning)"
        elif eval_cp >= 50:
            eval_label = f"🟡 +{eval_cp} cp (White slightly better)"
        elif eval_cp <= -50:
            eval_label = f"🟡 {eval_cp} cp (Black slightly better)"
        else:
            eval_label = f"⚪ {eval_cp:+d} cp (Even)"
        c2.metric("Stockfish eval", eval_label)
        c3.metric("Ply", r["snapshot_ply"])

        white_prob = eval_to_win_prob(r["stockfish_eval"])
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#7d8fa8;font-size:0.8rem;"
            "font-weight:600;margin-bottom:6px;'>Evaluation bar</div>",
            unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;height:28px;border-radius:8px;"
            f"overflow:hidden;border:1px solid #1f2b3d;'>"
            f"<div style='width:{white_prob:.1%};background:#22c55e;"
            f"display:flex;align-items:center;justify-content:center;"
            f"color:#052e16;font-weight:700;font-size:0.8rem;'>"
            f"{white_prob:.0%}</div>"
            f"<div style='width:{1-white_prob:.1%};background:#1a2332;"
            f"display:flex;align-items:center;justify-content:center;"
            f"color:#7d8fa8;font-weight:700;font-size:0.8rem;'>"
            f"{1-white_prob:.0%}</div>"
            f"</div>",
            unsafe_allow_html=True)

    # ---------- Card 2: Prediction ----------
    with st.container(border=True):
        st.markdown("### 🎯 Prediction")

        if r["p_white"] >= r["p_black"] and r["p_white"] >= r["p_draw"]:
            gauge_val = r["p_white"]
            gauge_label = "White to win"
            gauge_color = "#22c55e"
        elif r["p_black"] >= r["p_white"] and r["p_black"] >= r["p_draw"]:
            gauge_val = r["p_black"]
            gauge_label = "Black to win"
            gauge_color = "#3b82f6"
        else:
            gauge_val = r["p_draw"]
            gauge_label = "Draw expected"
            gauge_color = "#a855f7"

        g1, g2 = st.columns([1, 1.4])
        with g1:
            render_gauge(gauge_val, gauge_label, gauge_color)
        with g2:
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            render_probability_bars(r["p_white"], r["p_draw"], r["p_black"])
            st.markdown(
                f"<div style='text-align:center;color:#7d8fa8;"
                f"font-size:0.82rem;margin-top:10px;'>"
                f"Confidence: "
                f"<span style='color:#e6edf3;font-weight:600;'>"
                f"{r['confidence']:.1%}</span></div>",
                unsafe_allow_html=True)

    # ---------- Position details expander ----------
    with st.expander("Position details"):
        st.code(r["fen"], language="text")
        fen_url = r["fen"].replace(" ", "_")
        st.markdown(f"[Open on Lichess](https://lichess.org/analysis/standard/{fen_url})")


# ---------- UI ----------
st.set_page_config(page_title="Chess Predictor", page_icon="♟️", layout="wide")
# ---------- Dark card CSS ----------
import base64, os

@st.cache_data
def _img_as_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


_bg_b64 = _img_as_b64(get_app_path("static/bg-chess.png"))

st.markdown(f"""
<style>
/* Background image with navy overlay */
.stApp {{
    background:
        linear-gradient(90deg,
            rgba(5, 10, 25, 0.75) 0%,
            rgba(5, 10, 25, 0.55) 40%,
            rgba(10, 20, 45, 0.35) 65%,
            rgba(15, 25, 55, 0.20) 100%),
        url("data:image/png;base64,{_bg_b64}") no-repeat right center;
    background-size: cover, cover;
    background-attachment: fixed, fixed;
    background-color: #03060f;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at 75% 30%,
                rgba(80, 90, 200, 0.15) 0%,
                transparent 55%);
    pointer-events: none;
    z-index: 0;
}}

.main .block-container {{
    padding-top: 1.5rem;
    max-width: 1600px;
    position: relative;
    z-index: 1;
}}

/* Glass Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: rgba(10, 20, 45, 0.72);
    border: 1px solid rgba(60, 90, 160, 0.30);
    border-radius: 14px;
    padding: 8px 14px;
    box-shadow: 0 2px 14px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(12px);
}}
div[data-testid="stVerticalBlockBorderWrapper"] h3 {{
    color: #e6edf3 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}}

/* Buttons */
.stButton button {{
    background-color: rgba(20, 32, 60, 0.9);
    color: #e6edf3;
    border: 1px solid #27374f;
    border-radius: 8px;
    font-weight: 600;
}}
.stButton button:hover {{
    background-color: #1e2a4d;
    border-color: #3b82f6;
}}
.stButton button[kind="primary"] {{
    background: linear-gradient(135deg, #22c55e, #16a34a);
    border: 1px solid #22c55e;
    color: #ffffff;
    box-shadow: 0 0 14px rgba(34,197,94,0.4);
}}
.stButton button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #16a34a, #15803d);
    box-shadow: 0 0 20px rgba(34,197,94,0.6);
}}

/* Tabs */
.stTabs [role="tablist"] {{
    background: rgba(10, 20, 45, 0.9) !important;
    padding: 8px 12px !important;
    border-radius: 50px !important;
    border: 1px solid rgba(80, 120, 200, 0.4) !important;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.5), 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    display: inline-flex !important;
    gap: 8px !important;
    margin-bottom: 12px !important;
    width: auto !important;
}}
.stTabs [role="tab"] {{
    background: transparent !important;
    color: #8b9bb4 !important;
    border-radius: 50px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
}}
.stTabs [role="tab"]:hover {{
    color: #c9d1d9 !important;
    background: rgba(30, 50, 90, 0.5) !important;
}}
.stTabs [role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(59, 130, 246, 0.3)) !important;
    color: #22e66b !important;
    border: 1px solid rgba(34, 197, 94, 0.7) !important;
    box-shadow: 0 0 18px rgba(34, 197, 94, 0.5), inset 0 0 14px rgba(34, 197, 94, 0.2) !important;
    font-weight: 700 !important;
    filter: drop-shadow(0 0 8px rgba(34, 197, 94, 0.8)) !important;
    text-shadow: 0 0 10px rgba(34, 197, 94, 0.5) !important;
}}

/* Hide the red underline */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}

/* Text / Inputs */
h1 {{ font-weight: 800 !important; letter-spacing: -0.5px; }}
div[data-testid="stAlert"] {{
    background-color: rgba(10, 20, 45, 0.9);
    border: 1px solid #22c55e;
    border-left: 4px solid #22c55e;
    color: #e6edf3;
}}
div[data-testid="stMetricValue"] {{ color: #22c55e !important; font-weight: 700 !important; }}
div[data-testid="stMetricLabel"] {{ color: #7d8fa8 !important; }}

section[data-testid="stSidebar"] {{
    background-color: rgba(6, 10, 25, 0.97);
    border-right: 1px solid rgba(40, 70, 130, 0.4);
}}

/* Strip wrapper backgrounds */
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stTextInput"] > div > div > div,
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stNumberInput"] > div > div > div,
[data-testid="stTextArea"] > div,
[data-testid="stTextArea"] > div > div,
[data-testid="stTextArea"] > div > div > div {{
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
}}

/* Glassy fields */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {{
    background: rgba(15, 25, 50, 0.25) !important;
    border: 1px solid rgba(60, 90, 160, 0.4) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    backdrop-filter: blur(10px) !important;
    font-family: 'Consolas', 'Monaco', monospace !important;
    font-size: 0.85rem !important;
}}
[data-testid="stNumberInput"] button {{
    background: rgba(15, 25, 50, 0.5) !important;
    border: 1px solid rgba(60, 90, 160, 0.4) !important;
    color: #e6edf3 !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: rgba(15, 25, 50, 0.35) !important;
    border: 1px solid rgba(60, 90, 160, 0.4) !important;
    color: #e6edf3 !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
    border-color: rgba(34, 197, 94, 0.7) !important;
    box-shadow: 0 0 12px rgba(34, 197, 94, 0.35) !important;
}}

/* Plotly modebar */
[data-testid="stPlotlyChart"] .modebar {{
    top: auto !important;
    bottom: 6px !important;
    right: 6px !important;
    opacity: 1 !important;
    background: rgba(10, 20, 45, 0.9) !important;
    border: 1px solid rgba(96, 165, 250, 0.35) !important;
    border-radius: 8px !important;
    padding: 2px 4px !important;
    z-index: 10 !important;
}}
[data-testid="stPlotlyChart"] .modebar-btn svg {{ fill: #c9d1d9 !important; opacity: 1 !important; }}
[data-testid="stPlotlyChart"] .modebar-btn:hover svg {{ fill: #22c55e !important; }}

/* View mode radio — glowing pill toggle */
.st-key-view_mode_box [data-testid="stRadio"] > div {{
    gap: 10px !important;
    display: flex !important;
    flex-direction: row !important;
}}
.st-key-view_mode_box label {{
    background: rgba(10,20,45,0.65) !important;
    border: 1px solid rgba(96,165,250,0.35) !important;
    border-radius: 50px !important;
    padding: 10px 22px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    color: #c9d1d9 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}}
.st-key-view_mode_box label:hover {{
    background: rgba(20,35,70,0.85) !important;
    border-color: rgba(96,165,250,0.6) !important;
}}
.st-key-view_mode_box label > div:first-child {{ display: none !important; }}
.st-key-view_mode_box label:has(input:checked),
.st-key-view_mode_box label[aria-checked="true"] {{
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.55), inset 0 0 12px rgba(96,165,250,0.25) !important;
}}
.st-key-view_mode_box label[aria-checked="true"] * {{ color: #ffffff !important; }}

/* Side Filter Radio (All / White / Black) */
.st-key-side_filter_box_1 [data-testid="stRadio"] > div,
.st-key-side_filter_box_2 [data-testid="stRadio"] > div {{
    gap: 8px !important;
    display: flex !important;
    flex-direction: row !important;
}}
.st-key-side_filter_box_1 label,
.st-key-side_filter_box_2 label {{
    background: rgba(10,20,45,0.65) !important;
    border: 1px solid rgba(96,165,250,0.35) !important;
    border-radius: 50px !important;
    padding: 6px 16px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    color: #c9d1d9 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}}
.st-key-side_filter_box_1 label:hover,
.st-key-side_filter_box_2 label:hover {{
    background: rgba(20,35,70,0.85) !important;
    border-color: rgba(96,165,250,0.6) !important;
}}
.st-key-side_filter_box_1 label > div:first-child,
.st-key-side_filter_box_2 label > div:first-child {{ display: none !important; }}
.st-key-side_filter_box_1 label:has(input:checked),
.st-key-side_filter_box_1 label[aria-checked="true"],
.st-key-side_filter_box_2 label:has(input:checked),
.st-key-side_filter_box_2 label[aria-checked="true"] {{
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(59,130,246,0.55), inset 0 0 10px rgba(96,165,250,0.25) !important;
}}
.st-key-side_filter_box_1 label[aria-checked="true"] *,
.st-key-side_filter_box_2 label[aria-checked="true"] * {{ color: #ffffff !important; }}
</style>
""", unsafe_allow_html=True)
st.title("♟️ Chess Game Predictor")
st.markdown(
    "Three ways to predict: **PGN** (past game), **FEN** (any position), "
    "or the **Interactive Board** (drag pieces and analyze).")

tab_pgn, tab_fen, tab_board, tab_prematch, tab_review = st.tabs([
    "📄 PGN Input", "🎯 Position (FEN)", "🎨 Interactive Board",
    "⚔️ Pre-Match", "📊 Review"])


# ============ TAB 1 — PGN ============
with tab_pgn:
    col_pgn, col_info = st.columns([1.5, 1], gap="medium")

    with col_pgn:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.2rem;font-weight:700;"
                "color:#e6edf3;margin-bottom:8px;'>📄 PGN</div>",
                unsafe_allow_html=True)

            if st.button("Load sample game", key="load_pgn"):
                st.session_state["pgn"] = """[Event "Sample"]
[White "Speelman, Jonathan"]
[Black "Gulko, Boris"]
[Result "1-0"]
[WhiteElo "2615"]
[BlackElo "2610"]

1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 d6 6. Bg5 Qb6 7. Nb3 e6 8. Bf4 Ne5 9. Be3 Qc7 10. f4 Nc6 11. g4 d5 12. e5 Nd7 13. Nb5 Qd8 14. h4 f6 15. Nd6+ Bxd6 16. exd6 Nb6 17. g5 O-O 18. Qd2 Na4 19. O-O-O a6 20. g6 Qxd6 21. gxh7+ Kh8 22. h5 Bd7 23. Bd3 d4 24. h6 gxh6 25. Qg2 Nb4 26. Rdg1 Nxd3+ 27. Kb1 1-0"""

            pgn_input = st.text_area(
                "PGN", height=220, key="pgn", label_visibility="collapsed",
                placeholder="Paste a PGN here...")

    with col_info:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.2rem;font-weight:700;"
                "color:#e6edf3;margin-bottom:12px;'>👥 Game info</div>",
                unsafe_allow_html=True)

            # Try to parse the current PGN for header info
            info_rows = [
                ("👤", "White", "—"),
                ("👤", "Black", "—"),
                ("🏆", "Result", "—"),
                ("♟️", "Moves", "—"),
            ]
            if pgn_input.strip():
                try:
                    import chess.pgn as _pgn, io as _io
                    _g = _pgn.read_game(_io.StringIO(pgn_input))
                    if _g is not None:
                        _h = _g.headers
                        _w = _h.get("White", "?")
                        _we = _h.get("WhiteElo", "")
                        _b = _h.get("Black", "?")
                        _be = _h.get("BlackElo", "")
                        _r = _h.get("Result", "—")
                        _n = len(list(_g.mainline_moves()))
                        info_rows = [
                            ("👤", "White", f"{_w}" + (f" ({_we})" if _we else "")),
                            ("👤", "Black", f"{_b}" + (f" ({_be})" if _be else "")),
                            ("🏆", "Result", _r),
                            ("♟️", "Moves", str(_n)),
                        ]
                except Exception:
                    pass

            rows_html = ""
            for icon, k, v in info_rows:
                rows_html += (
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;padding:10px 0;"
                    f"border-bottom:1px solid #1a2332;'>"
                    f"<span style='color:#7d8fa8;font-size:0.85rem;"
                    f"display:flex;align-items:center;gap:8px;'>"
                    f"<span>{icon}</span>{k}</span>"
                    f"<span style='color:#e6edf3;font-size:0.9rem;"
                    f"font-weight:600;'>{v}</span>"
                    f"</div>"
                )
            st.markdown(rows_html, unsafe_allow_html=True)

    if st.button("⚡ Predict", type="primary", key="predict_pgn"):
        if not pgn_input.strip():
            st.warning("Please paste a PGN first.")
        else:
            with st.spinner("Analyzing..."):
                r = predict_from_pgn(pgn_input)
            if "error" in r:
                st.error(r["error"])
            else:
                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.markdown("### 👥 Players")
                        st.write(f"**White:** {r['white']} ({r['white_elo']})")
                        st.write(f"**Black:** {r['black']} ({r['black_elo']})")
                        st.write(f"**Snapshot:** after ply {r['snapshot_ply']}")
                with c2:
                    with st.container(border=True):
                        st.markdown("### 📊 Position analysis")
                        st.write(f"**Material balance:** {r['material_balance']:+d}")
                        st.write(f"**Stockfish eval:** {r['stockfish_eval']:+d} cp")

                with st.container(border=True):
                    st.markdown("### 🎯 Prediction")
                    if r["p_white"] >= r["p_black"] and r["p_white"] >= r["p_draw"]:
                        gauge_val = r["p_white"]
                        gauge_label = "White to win"
                        gauge_color = "#22c55e"
                    elif r["p_black"] >= r["p_white"] and r["p_black"] >= r["p_draw"]:
                        gauge_val = r["p_black"]
                        gauge_label = "Black to win"
                        gauge_color = "#3b82f6"
                    else:
                        gauge_val = r["p_draw"]
                        gauge_label = "Draw expected"
                        gauge_color = "#8b949e"

                    g1, g2 = st.columns([1, 1.4])
                    with g1:
                        render_gauge(gauge_val, gauge_label, gauge_color)
                    with g2:
                        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                        render_probability_bars(r["p_white"], r["p_draw"], r["p_black"])

                with st.expander("View FEN"):
                    st.code(r["fen"])
                open_name, open_eco = detect_opening(pgn_input)
                if open_name:
                    st.info(f"📖 Opening: **{open_name}** (ECO {open_eco})")
                with st.expander("Copy PGN"):
                    st.code(pgn_input, language="text")
                    st.caption("Use the copy icon in the top-right of the code box above.")
            # ---- Move-by-move analysis ----
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    hdr_left, hdr_right = st.columns([1.4, 1])
    with hdr_left:
        st.markdown(
            "<div style='font-size:1.6rem;font-weight:800;"
            "color:#e6edf3;letter-spacing:-0.5px;'>"
            "📊 Move-by-Move Analysis</div>"
            "<div style='color:#7d8fa8;font-size:0.9rem;margin-top:2px;'>"
            "Explore every move, see the evaluation, and identify the best, "
            "good, inaccurate, and blunder moves.</div>",
            unsafe_allow_html=True)
    with hdr_right:
        st.markdown(
            "<div style='display:flex;gap:6px;justify-content:flex-end;"
            "flex-wrap:wrap;margin-top:8px;'>"
            "<span style='background:#1e2a3d;color:#facc15;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>⭐ Best</span>"
            "<span style='background:#1e2a3d;color:#22c55e;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>🌟 Excellent</span>"
            "<span style='background:#1e2a3d;color:#60a5fa;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>✅ Good</span>"
            "<span style='background:#1e2a3d;color:#f97316;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>⚠️ Inaccuracy</span>"
            "<span style='background:#1e2a3d;color:#ef4444;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>❌ Mistake</span>"
            "<span style='background:#1e2a3d;color:#ec4899;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>💥 Blunder</span>"
            "</div>",
            unsafe_allow_html=True)
    # Trigger analysis (only once)
    if st.button("🔍 Analyze every move", key="analyze_moves", type="primary"):
        if not pgn_input.strip():
            st.warning("Please paste a PGN first.")
        else:
            with st.spinner("Analyzing every move... (~15-30 seconds)"):
                st.session_state["move_analysis"] = analyze_game_moves(pgn_input)

    # Render results from session_state (persists across reruns)
    move_results = st.session_state.get("move_analysis")
    if move_results is not None:
        if "error" in move_results:
            st.error(move_results["error"])
        else:
            moves = move_results["moves"]
            from collections import Counter
            w_counts = Counter(m["category"] for m in moves if m["color"] == "White")
            b_counts = Counter(m["category"] for m in moves if m["color"] == "Black")
            categories = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
            emojis = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                      "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}

            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("### ⚪ White move quality")
                    for cat in categories:
                        st.write(f"{emojis[cat]} {cat}: **{w_counts.get(cat, 0)}**")
            with c2:
                with st.container(border=True):
                    st.markdown("### ⚫ Black move quality")
                    for cat in categories:
                        st.write(f"{emojis[cat]} {cat}: **{b_counts.get(cat, 0)}**")

            with st.container(border=True):
                st.markdown(
                    "<div style='font-size:1.3rem;font-weight:800;"
                    "color:#e6edf3;margin-bottom:10px;'>"
                    "📋 All Moves</div>",
                    unsafe_allow_html=True)

                ctrl1, ctrl2, ctrl3 = st.columns([1.8, 1.5, 1.2])
                with ctrl1:
                    search_q = st.text_input(
                        "Search", placeholder="🔍 Search move (e.g. e4, Nf3...)",
                        label_visibility="collapsed", key="mv_search")
                with ctrl2:
                    with st.container(key="side_filter_box_1"):
                        side_filter = st.radio(
                            "Side", ["All", "White", "Black"],
                            horizontal=True, label_visibility="collapsed",
                            key="mv_side")
                with ctrl3:
                    sort_by = st.selectbox(
                        "Sort", ["Move number", "CPL (high→low)", "CPL (low→high)"],
                        label_visibility="collapsed", key="mv_sort")

                filtered = list(moves)
                if side_filter != "All":
                    filtered = [m for m in filtered if m["color"] == side_filter]
                if search_q.strip():
                    q = search_q.strip().lower()
                    filtered = [m for m in filtered if q in m["san"].lower()]
                if sort_by == "CPL (high→low)":
                    filtered = sorted(filtered, key=lambda m: -m["cpl"])
                elif sort_by == "CPL (low→high)":
                    filtered = sorted(filtered, key=lambda m: m["cpl"])

                pill_colors = {
                    "Best": ("#facc15", "#422006"),
                    "Excellent": ("#22c55e", "#052e16"),
                    "Good": ("#60a5fa", "#0c1e3a"),
                    "Inaccuracy": ("#f97316", "#3a1a03"),
                    "Mistake": ("#ef4444", "#3a0d0d"),
                    "Blunder": ("#ec4899", "#3a0d24"),
                }
                emoji_map = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                             "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}

                rows_html = ""
                for idx, m in enumerate(filtered):
                    cat = m["category"]
                    fg, bg = pill_colors.get(cat, ("#8b949e", "#1e2a3d"))
                    emoji = emoji_map.get(cat, "")
                    eval_cp = m.get("eval_after", 0)
                    eval_color = "#22c55e" if eval_cp >= 0 else "#ef4444"
                    side_dot = "#ffffff" if m["color"] == "White" else "#4b5563"
                    row_bg = "#0d1420" if idx % 2 == 0 else "#0a0f16"
                    rows_html += (
                        f"<tr style='background:{row_bg};border-bottom:1px solid #1a2332;'>"
                        f"<td style='padding:10px 12px;color:#7d8fa8;font-weight:600;'>{m['move_number']}</td>"
                        f"<td style='padding:10px 12px;color:#e6edf3;font-weight:700;'>{m['san']}</td>"
                        f"<td style='padding:10px 12px;color:#c9d1d9;'>"
                        f"<span style='display:inline-flex;align-items:center;gap:6px;'>"
                        f"<span style='width:8px;height:8px;border-radius:50%;background:{side_dot};'></span>"
                        f"{m['color']}</span></td>"
                        f"<td style='padding:10px 12px;color:{eval_color};font-weight:600;'>{eval_cp:+d}</td>"
                        f"<td style='padding:10px 12px;color:#7d8fa8;'>{m['cpl']}</td>"
                        f"<td style='padding:10px 12px;'>"
                        f"<span style='background:{bg};color:{fg};padding:4px 12px;"
                        f"border-radius:14px;font-size:0.8rem;font-weight:700;'>"
                        f"{emoji} {cat}</span></td>"
                        f"<td style='padding:10px 12px;color:#7d8fa8;text-align:right;'>›</td>"
                        f"</tr>"
                    )

                table_html = (
                    "<div style='overflow-x:auto;border-radius:10px;margin-top:8px;'>"
                    "<table style='width:100%;border-collapse:collapse;font-size:0.9rem;"
                    "font-family:-apple-system,BlinkMacSystemFont,sans-serif;'>"
                    "<thead><tr style='background:#0a0f16;border-bottom:2px solid #1f2b3d;'>"
                    "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>#</th>"
                    "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>Move</th>"
                    "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>Side</th>"
                    "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>Eval (cp)</th>"
                    "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>CPL</th>"
                    "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>Rating</th>"
                    "<th style='padding:10px 12px;'></th>"
                    "</tr></thead>"
                    f"<tbody>{rows_html}</tbody>"
                    "</table></div>"
                )
                st.markdown(table_html, unsafe_allow_html=True)
                st.caption(f"Showing {len(filtered)} of {len(moves)} moves")

            if st.button("🗑️ Clear Analysis", key="clear_analysis"):
                st.session_state.pop("move_analysis", None)
                st.rerun()

                key = [m for m in moves if m["category"] in ("Mistake", "Blunder")]
                if key:
                    with st.container(border=True):
                        st.markdown("### 🔴 Key mistakes and blunders")
                        for m in key:
                            st.write(f"- Move {m['move_number']} ({m['color']}): "
                                     f"**{m['san']}** → {m['label']} (lost **{m['cpl']}** cp)")
                else:
                    st.success("No mistakes or blunders detected! 🎉")


# ============ TAB 2 — FEN ============
with tab_fen:
    st.markdown("Paste a **FEN** string. Use [lichess.org/analysis](https://lichess.org/analysis) to set up positions visually and copy the FEN.")
    fen_input = st.text_input("FEN:", value="r4r1k/1p1b3P/p2qpp1p/8/n2p1P2/1N1nB3/PPP3Q1/1K4RR b - - 1 27", key="fen")
    col_w, col_b = st.columns(2)
    with col_w:
        white_elo = st.number_input("White ELO:", 0, 3500, 2615, 10, key="w_elo")
    with col_b:
        black_elo = st.number_input("Black ELO:", 0, 3500, 2610, 10, key="b_elo")
    if st.button("Predict", type="primary", key="predict_fen"):
        if not fen_input.strip():
            st.warning("Please paste a FEN first.")
        else:
            with st.spinner("Analyzing position..."):
                r = predict_from_fen(fen_input, int(white_elo), int(black_elo))
            render_analysis(r)


# ============ TAB 3 — Interactive Board ============
with tab_board:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:14px;margin-bottom:14px;'>"
        "<span style='font-size:2.2rem;'>♞</span>"
        "<div>"
        "<div style='font-size:1.8rem;font-weight:800;color:#e6edf3;"
        "letter-spacing:-0.5px;'>Chess Game Setup</div>"
        "<div style='color:#7d8fa8;font-size:0.95rem;'>"
        "Drag pieces to set up any position. The prediction updates automatically."
        "</div></div></div>",
        unsafe_allow_html=True)

    if "board_fen" not in st.session_state:
        st.session_state.board_fen = START_FEN
    if "board_analysis" not in st.session_state:
        st.session_state.board_analysis = None
    if "analyzed_fen" not in st.session_state:
        st.session_state.analyzed_fen = None
    if "analyzed_elo" not in st.session_state:
        st.session_state.analyzed_elo = None
    if "board_orientation" not in st.session_state:
        st.session_state.board_orientation = "white"
    if "just_reset" not in st.session_state:
        st.session_state.just_reset = False
    if "game_pgn" not in st.session_state:
        st.session_state.game_pgn = ""
    if "moves_list" not in st.session_state:
        st.session_state.moves_list = []
    if "last_move_highlight" not in st.session_state:
        st.session_state.last_move_highlight = None
    if "current_view_index" not in st.session_state:
        st.session_state.current_view_index = -1
    if "board_theme" not in st.session_state:
        st.session_state.board_theme = "classic"

    data = bridge("my-bridge")
    if data and isinstance(data, dict) and "fen" in data:
        if st.session_state.just_reset:
            st.session_state.just_reset = False
        else:
            new_fen = data["fen"]
            if new_fen != st.session_state.board_fen:
                st.session_state.board_fen = new_fen
                st.session_state.board_analysis = None
                st.session_state.best_move = None
                history = data.get("history")
                if isinstance(history, str):
                    try:
                        history = json.loads(history)
                    except Exception:
                        history = None
                if isinstance(history, list) and history:
                    st.session_state.moves_list = [
                        (m.get("color", "w"), m.get("san", ""),
                         m.get("fen", ""), m.get("from", "") + m.get("to", ""))
                        for m in history]
                    st.session_state.current_view_index = len(st.session_state.moves_list) - 1
                    last = history[-1]
                    st.session_state.last_move_highlight = last.get("from", "") + last.get("to", "")
                    st.session_state.game_pgn = _rebuild_pgn(st.session_state.moves_list)

    elo_left, elo_right = st.columns(2, gap="medium")

    with elo_left:
        with st.container(border=True):
            st.markdown(
                "<div style='display:flex;align-items:center;gap:8px;"
                "margin-bottom:6px;'>"
                "<span style='font-size:1.1rem;'>👤</span>"
                "<span style='color:#c9d1d9;font-weight:600;'>White ELO</span>"
                "</div>",
                unsafe_allow_html=True)
            board_white_elo = st.number_input(
                "White ELO", 0, 3500, 2615, 10,
                key="board_w_elo", label_visibility="collapsed")

    with elo_right:
        with st.container(border=True):
            st.markdown(
                "<div style='display:flex;align-items:center;gap:8px;"
                "margin-bottom:6px;'>"
                "<span style='font-size:1.1rem;'>👤</span>"
                "<span style='color:#c9d1d9;font-weight:600;'>Black ELO</span>"
                "</div>",
                unsafe_allow_html=True)
            board_black_elo = st.number_input(
                "Black ELO", 0, 3500, 2610, 10,
                key="board_b_elo", label_visibility="collapsed")

    opt_c1, opt_c2, opt_c3 = st.columns([2, 1.3, 1])
    with opt_c1:
        auto_analyze = st.checkbox("Auto-analyze after each move", value=False, key="auto_analyze")
    with opt_c2:
        if st.button("🔄 Flip Board", key="flip_board", use_container_width=True):
            st.session_state.board_orientation = ("black" if st.session_state.board_orientation == "white" else "white")
            st.rerun()
    with opt_c3:
        st.markdown(
            f"<div style='padding-top:8px;color:#7d8fa8;font-size:0.9rem;'>"
            f"👁️ View: <b style='color:#e6edf3;'>"
            f"{st.session_state.board_orientation.capitalize()}</b></div>",
            unsafe_allow_html=True)

    theme_col1, theme_col2 = st.columns([1, 1])
    with theme_col1:
        st.selectbox("🎨 Board theme:", ["classic", "green", "blue", "gray", "purple"], key="board_theme")

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    current_elo = (int(board_white_elo), int(board_black_elo))
    needs_analysis = (
        st.session_state.analyzed_fen != st.session_state.board_fen
        or st.session_state.analyzed_elo != current_elo)

    if auto_analyze and needs_analysis:
        with st.spinner("Analyzing position..."):
            st.session_state.board_analysis = predict_from_fen(
                st.session_state.board_fen, current_elo[0], current_elo[1])
            st.session_state.analyzed_fen = st.session_state.board_fen
            st.session_state.analyzed_elo = current_elo
    board_col, side_col = st.columns([1.6, 1], gap="medium")

    with board_col:
        with st.container(border=True):
            bm = st.session_state.get("best_move")
            arrow_uci = bm.get("uci") if isinstance(bm, dict) else None
            board_render = Chess(
                400,
                st.session_state.board_fen,
                orientation=st.session_state.board_orientation,
                highlight_move=st.session_state.last_move_highlight,
                move_history=[
                    {"san": m[1], "from": m[3][:2], "to": m[3][2:4],
                     "color": m[0], "fen": m[2]}
                    for m in st.session_state.moves_list],
                theme=st.session_state.board_theme,
                arrow_move=arrow_uci,
            )
            components.html(board_render.puzzle_board(), height=720, scrolling=False)

    with side_col:
        # ---- Stats card ----
        with st.container(border=True):
            st.markdown("### ⚡ Stats")
            stat_c1, stat_c2 = st.columns(2)
            with stat_c1:
                st.markdown(
                    "<div style='color:#7d8fa8;font-size:0.8rem;"
                    "font-weight:600;margin-bottom:4px;'>Moves</div>"
                    f"<div style='color:#3b82f6;font-size:1.8rem;"
                    f"font-weight:800;line-height:1;'>"
                    f"{len(st.session_state.moves_list)}</div>",
                    unsafe_allow_html=True)
            with stat_c2:
                try:
                    side_to_move = "White" if chess.Board(
                        st.session_state.board_fen).turn else "Black"
                except Exception:
                    side_to_move = "White"
                st.markdown(
                    "<div style='color:#7d8fa8;font-size:0.8rem;"
                    "font-weight:600;margin-bottom:4px;'>Status</div>"
                    f"<div style='color:#22c55e;font-size:0.95rem;"
                    f"font-weight:700;padding-top:6px;'>"
                    f"● {side_to_move} to move</div>",
                    unsafe_allow_html=True)

        # ---- Prediction card ----
        with st.container(border=True):
            st.markdown("### 🧠 Prediction")
            ra = st.session_state.get("board_analysis")
            if ra and "error" not in ra:
                render_probability_bars(ra["p_white"], ra["p_draw"], ra["p_black"])
                st.markdown(
                    f"<div style='text-align:center;color:#7d8fa8;"
                    f"font-size:0.82rem;margin-top:8px;'>"
                    f"<b style='color:#e6edf3;'>{ra['label']}</b> · "
                    f"{ra['confidence']:.0%} confidence</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='color:#7d8fa8;font-size:0.85rem;"
                    "padding:20px 0;text-align:center;'>"
                    "Make a move to see the prediction...</div>",
                    unsafe_allow_html=True)

        # ---- Moves this session (big blue button style) ----
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#2563eb,#7c3aed);"
            f"color:white;padding:14px;border-radius:10px;text-align:center;"
            f"font-weight:700;margin-bottom:8px;"
            f"box-shadow:0 4px 16px rgba(59,130,246,0.35);'>"
            f"♟️ Moves this session: {len(st.session_state.moves_list)}</div>",
            unsafe_allow_html=True)

        # ---- Undo last move ----
        if st.button("↺ Undo Last Move", use_container_width=True, key="undo_last"):
            if st.session_state.moves_list:
                st.session_state.moves_list.pop()
                if st.session_state.moves_list:
                    last = st.session_state.moves_list[-1]
                    st.session_state.board_fen = last[2]
                    st.session_state.last_move_highlight = last[3]
                    st.session_state.current_view_index = len(st.session_state.moves_list) - 1
                else:
                    st.session_state.board_fen = START_FEN
                    st.session_state.last_move_highlight = None
                    st.session_state.current_view_index = -1
                st.session_state.just_reset = True
                st.session_state.board_analysis = None
                st.rerun()



    if not auto_analyze:
        if st.button("🔍 Analyze Position", type="primary", key="manual_analyze", use_container_width=True):
            with st.spinner("Analyzing position..."):
                st.session_state.board_analysis = predict_from_fen(
                    st.session_state.board_fen, int(board_white_elo), int(board_black_elo))
                st.session_state.analyzed_fen = st.session_state.board_fen
                st.session_state.analyzed_elo = (int(board_white_elo), int(board_black_elo))

    bm_col1, bm_col2 = st.columns(2)
    with bm_col1:
        if st.button("💡 Show Best Move", key="best_move_btn", use_container_width=True):
            with st.spinner("Asking Stockfish..."):
                try:
                    temp_board = chess.Board(st.session_state.board_fen)
                    st.session_state.best_move = stockfish_best_move(temp_board)
                except Exception as e:
                    st.session_state.best_move = {"error": str(e)}
            st.rerun()
    with bm_col2:
        if st.button("🔄 Reset to Starting Position", key="reset_board", use_container_width=True):
            st.session_state.board_fen = START_FEN
            st.session_state.board_analysis = None
            st.session_state.analyzed_fen = None
            st.session_state.analyzed_elo = None
            st.session_state.best_move = None
            st.session_state.game_pgn = ""
            st.session_state.moves_list = []
            st.session_state.last_move_highlight = None
            st.session_state.just_reset = True
            st.rerun()

    if st.session_state.get("best_move"):
        bm = st.session_state.best_move
        if "error" in bm:
            st.error(f"Could not compute best move: {bm['error']}")
        else:
            side = "White" if chess.Board(st.session_state.board_fen).turn else "Black"
            st.success(f"💡 **Best move for {side}: {bm['san']}**  (coordinate: `{bm['uci']}`)")

    st.caption(f"Current FEN: `{st.session_state.board_fen}`")

    if st.session_state.board_analysis:
        render_analysis(st.session_state.board_analysis)
    else:
        if not auto_analyze:
            st.info("Click **Analyze** above to run a prediction.")

    # Move History Panel
    with st.expander("📜 Move History (click any move to jump)", expanded=False):
        if not st.session_state.moves_list:
            st.caption("No moves yet — drag pieces on the board.")
        else:
            moves = st.session_state.moves_list
            pairs = []
            for i in range(0, len(moves), 2):
                num = i // 2 + 1
                w = moves[i]
                b = moves[i+1] if i+1 < len(moves) else None
                pairs.append((num, w, b, i))
            for num, w, b, base_idx in pairs:
                cols = st.columns([1, 2, 2])
                cols[0].markdown(f"**{num}.**")
                if cols[1].button(w[1], key=f"mv_w_{base_idx}", use_container_width=True):
                    st.session_state.board_fen = w[2]
                    st.session_state.last_move_highlight = w[3]
                    st.session_state.current_view_index = base_idx
                    st.session_state.board_analysis = None
                    st.session_state.just_reset = True
                    st.rerun()
                if b is not None:
                    if cols[2].button(b[1], key=f"mv_b_{base_idx+1}", use_container_width=True):
                        st.session_state.board_fen = b[2]
                        st.session_state.last_move_highlight = b[3]
                        st.session_state.current_view_index = base_idx + 1
                        st.session_state.board_analysis = None
                        st.session_state.just_reset = True
                        st.rerun()

    # Copy PGN / Send to Review
    st.divider()
    if st.button("📋 Copy Current PGN", key="copy_board_pgn", use_container_width=True):
        pgn_to_copy = st.session_state.get("game_pgn", "").strip()
        if pgn_to_copy:
            st.code(pgn_to_copy, language="text")
            st.caption("Use the copy icon in the top-right of the code box above.")
        else:
            st.warning("No moves yet — play some moves first.")

    if st.button("📊 Send Game to Review Tab", key="send_to_review", use_container_width=True):
        pgn_to_send = st.session_state.get("game_pgn", "").strip()
        if not pgn_to_send:
            st.warning("Play some moves first, then send the game for review.")
        else:
            st.session_state["pending_review_pgn"] = pgn_to_send
            st.success("✅ Game sent! Switch to the **📊 Review** tab and click Generate Review.")

    # ---- Review moves ----
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    hdr_left, hdr_right = st.columns([1.4, 1])
    with hdr_left:
        st.markdown(
            "<div style='font-size:1.6rem;font-weight:800;"
            "color:#e6edf3;letter-spacing:-0.5px;'>"
            "📊 Review Your Moves</div>"
            "<div style='color:#7d8fa8;font-size:0.9rem;margin-top:2px;'>"
            "Play some moves on the board, then click below to rate each one.</div>",
            unsafe_allow_html=True)
    with hdr_right:
        st.markdown(
            "<div style='display:flex;gap:6px;justify-content:flex-end;"
            "flex-wrap:wrap;margin-top:8px;'>"
            "<span style='background:#1e2a3d;color:#facc15;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>⭐ Best</span>"
            "<span style='background:#1e2a3d;color:#22c55e;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>🌟 Excellent</span>"
            "<span style='background:#1e2a3d;color:#60a5fa;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>✅ Good</span>"
            "<span style='background:#1e2a3d;color:#f97316;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>⚠️ Inaccuracy</span>"
            "<span style='background:#1e2a3d;color:#ef4444;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>❌ Mistake</span>"
            "<span style='background:#1e2a3d;color:#ec4899;padding:4px 10px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600;'>💥 Blunder</span>"
            "</div>",
            unsafe_allow_html=True)

    pgn_preview = st.session_state.get("game_pgn", "")
    if pgn_preview:
        st.caption(f"Moves recorded: `{pgn_preview[:120]}{'...' if len(pgn_preview) > 120 else ''}`")
    else:
        st.info("No moves yet. Drag some pieces on the board above.")

    if st.button("🔍 Review My Moves", key="review_moves", type="primary"):
        pgn_text = st.session_state.get("game_pgn", "").strip()
        if not pgn_text:
            st.warning("Make a few moves first.")
        else:
            with st.spinner("Rating your moves... (~15-30 seconds)"):
                st.session_state["board_review"] = analyze_game_moves(
                    pgn_text, max_plies=80, depth=10)

    review_results = st.session_state.get("board_review")
    if review_results is not None:
        if "error" in review_results:
            st.error(f"Could not analyze: {review_results['error']}")
        else:
            moves = review_results["moves"]
            if not moves:
                st.warning("No moves found in your game.")
            else:
                from collections import Counter
                w_counts = Counter(m["category"] for m in moves if m["color"] == "White")
                b_counts = Counter(m["category"] for m in moves if m["color"] == "Black")
                categories = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
                emojis = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                          "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}

                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.markdown("### ⚪ White move quality")
                        for cat in categories:
                            st.write(f"{emojis[cat]} {cat}: **{w_counts.get(cat, 0)}**")
                with c2:
                    with st.container(border=True):
                        st.markdown("### ⚫ Black move quality")
                        for cat in categories:
                            st.write(f"{emojis[cat]} {cat}: **{b_counts.get(cat, 0)}**")

                with st.container(border=True):
                    st.markdown(
                        "<div style='font-size:1.3rem;font-weight:800;"
                        "color:#e6edf3;margin-bottom:10px;'>"
                        "📋 All Moves</div>",
                        unsafe_allow_html=True)

                    ctrl1, ctrl2, ctrl3 = st.columns([1.8, 1.5, 1.2])
                    with ctrl1:
                        search_q = st.text_input(
                            "Search", placeholder="🔍 Search move (e.g. e4, Nf3...)",
                            label_visibility="collapsed", key="bmv_search")
                    with ctrl2:
                        with st.container(key="side_filter_box_2"):
                            side_filter = st.radio(
                                "Side", ["All", "White", "Black"],
                                horizontal=True, label_visibility="collapsed",
                                key="bmv_side")
                    with ctrl3:
                        sort_by = st.selectbox(
                            "Sort", ["Move number", "CPL (high→low)", "CPL (low→high)"],
                            label_visibility="collapsed", key="bmv_sort")

                    filtered = list(moves)
                    if side_filter != "All":
                        filtered = [m for m in filtered if m["color"] == side_filter]
                    if search_q.strip():
                        q = search_q.strip().lower()
                        filtered = [m for m in filtered if q in m["san"].lower()]
                    if sort_by == "CPL (high→low)":
                        filtered = sorted(filtered, key=lambda m: -m["cpl"])
                    elif sort_by == "CPL (low→high)":
                        filtered = sorted(filtered, key=lambda m: m["cpl"])

                    pill_colors = {
                        "Best": ("#facc15", "#422006"),
                        "Excellent": ("#22c55e", "#052e16"),
                        "Good": ("#60a5fa", "#0c1e3a"),
                        "Inaccuracy": ("#f97316", "#3a1a03"),
                        "Mistake": ("#ef4444", "#3a0d0d"),
                        "Blunder": ("#ec4899", "#3a0d24"),
                    }
                    emoji_map = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                                 "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}

                    rows_html = ""
                    for idx, m in enumerate(filtered):
                        cat = m["category"]
                        fg, bg = pill_colors.get(cat, ("#8b949e", "#1e2a3d"))
                        emoji = emoji_map.get(cat, "")
                        eval_cp = m.get("eval_after", 0)
                        eval_color = "#22c55e" if eval_cp >= 0 else "#ef4444"
                        side_dot = "#ffffff" if m["color"] == "White" else "#4b5563"
                        row_bg = "#0d1420" if idx % 2 == 0 else "#0a0f16"
                        rows_html += (
                            f"<tr style='background:{row_bg};border-bottom:1px solid #1a2332;'>"
                            f"<td style='padding:10px 12px;color:#7d8fa8;font-weight:600;'>{m['move_number']}</td>"
                            f"<td style='padding:10px 12px;color:#e6edf3;font-weight:700;'>{m['san']}</td>"
                            f"<td style='padding:10px 12px;color:#c9d1d9;'>"
                            f"<span style='display:inline-flex;align-items:center;gap:6px;'>"
                            f"<span style='width:8px;height:8px;border-radius:50%;background:{side_dot};'></span>"
                            f"{m['color']}</span></td>"
                            f"<td style='padding:10px 12px;color:{eval_color};font-weight:600;'>{eval_cp:+d}</td>"
                            f"<td style='padding:10px 12px;color:#7d8fa8;'>{m['cpl']}</td>"
                            f"<td style='padding:10px 12px;'>"
                            f"<span style='background:{bg};color:{fg};padding:4px 12px;"
                            f"border-radius:14px;font-size:0.8rem;font-weight:700;'>"
                            f"{emoji} {cat}</span></td>"
                            f"<td style='padding:10px 12px;color:#7d8fa8;text-align:right;'>›</td>"
                            f"</tr>"
                        )

                    table_html = (
                        "<div style='overflow-x:auto;border-radius:10px;margin-top:8px;'>"
                        "<table style='width:100%;border-collapse:collapse;font-size:0.9rem;"
                        "font-family:-apple-system,BlinkMacSystemFont,sans-serif;'>"
                        "<thead><tr style='background:#0a0f16;border-bottom:2px solid #1f2b3d;'>"
                        "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>#</th>"
                        "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>Move</th>"
                        "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>Side</th>"
                        "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>Eval (cp)</th>"
                        "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>CPL</th>"
                        "<th style='padding:10px 12px;color:#7d8fa8;text-align:left;font-weight:600;font-size:0.75rem;text-transform:uppercase;'>Rating</th>"
                        "<th style='padding:10px 12px;'></th>"
                        "</tr></thead>"
                        f"<tbody>{rows_html}</tbody>"
                        "</table></div>"
                    )
                    st.markdown(table_html, unsafe_allow_html=True)
                    st.caption(f"Showing {len(filtered)} of {len(moves)} moves")

                key = [m for m in moves if m["category"] in ("Mistake", "Blunder")]
                if key:
                    with st.container(border=True):
                        st.markdown("### 🔴 Key mistakes and blunders")
                        for m in key:
                            st.write(f"- Move {m['move_number']} ({m['color']}): "
                                     f"**{m['san']}** → {m['label']} (lost **{m['cpl']}** cp)")
                else:
                    st.success("No mistakes or blunders detected! 🎉")


# ============ TAB 4 — Pre-Match ============
with tab_prematch:
    # ---- Header ----
    st.markdown(
        "<div style='display:flex;align-items:center;gap:14px;margin-bottom:16px;'>"
        "<span style='font-size:2.2rem;'>⚔️</span>"
        "<div>"
        "<div style='font-size:1.8rem;font-weight:800;color:#e6edf3;"
        "letter-spacing:-0.5px;'>Pre-Match Prediction</div>"
        "<div style='color:#7d8fa8;font-size:0.95rem;'>"
        "Predict the outcome before the game starts. Compare two models side by side."
        "</div></div></div>",
        unsafe_allow_html=True)

    # ---- ELO input cards ----
    elo_left, elo_right = st.columns(2, gap="medium")

    with elo_left:
        with st.container(border=True):
            st.markdown(
                "<div style='display:flex;align-items:center;gap:8px;"
                "margin-bottom:6px;'>"
                "<span style='font-size:1.1rem;'>👤</span>"
                "<span style='color:#c9d1d9;font-weight:600;'>White player ELO</span>"
                "</div>",
                unsafe_allow_html=True)
            pm_white_elo = st.number_input(
                "White ELO", 0, 3500, 2650, 10,
                key="pm_w_elo", label_visibility="collapsed")

    with elo_right:
        with st.container(border=True):
            st.markdown(
                "<div style='display:flex;align-items:center;gap:8px;"
                "margin-bottom:6px;'>"
                "<span style='font-size:1.1rem;'>👤</span>"
                "<span style='color:#c9d1d9;font-weight:600;'>Black player ELO</span>"
                "</div>",
                unsafe_allow_html=True)
            pm_black_elo = st.number_input(
                "Black ELO", 0, 3500, 2650, 10,
                key="pm_b_elo", label_visibility="collapsed")

    if st.button("⚔️ Predict Match Outcome", type="primary", key="pm_predict"):
        st.session_state["pm_results"] = predict_prematch_both(
            int(pm_white_elo), int(pm_black_elo))

    pm_results = st.session_state.get("pm_results")
    if pm_results:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;"
            f"margin-bottom:16px;'>"
            f"<span style='color:#7d8fa8;font-size:0.9rem;"
            f"text-transform:uppercase;letter-spacing:1px;font-weight:600;'>"
            f"ELO gap</span>"
            f"<span style='color:#e6edf3;font-size:1.4rem;font-weight:800;'>"
            f"{int(pm_white_elo) - int(pm_black_elo):+d}</span>"
            f"</div>",
            unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")

        # ---- Card 1: Accurate model ----
        with c1:
            with st.container(border=True):
                r = pm_results["unbalanced"]
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:10px;"
                    "margin-bottom:4px;'>"
                    "<span style='font-size:1.4rem;'>🎯</span>"
                    "<span style='font-size:1.2rem;font-weight:800;"
                    "color:#e6edf3;'>Accurate model</span></div>"
                    "<div style='color:#7d8fa8;font-size:0.82rem;"
                    "margin-bottom:14px;'>unbalanced · real class frequencies</div>",
                    unsafe_allow_html=True)

                # Donut
                if r["p_white"] >= r["p_black"] and r["p_white"] >= r["p_draw"]:
                    gv, gl, gc = r["p_white"], "White wins", "#22c55e"
                elif r["p_black"] >= r["p_white"] and r["p_black"] >= r["p_draw"]:
                    gv, gl, gc = r["p_black"], "Black wins", "#3b82f6"
                else:
                    gv, gl, gc = r["p_draw"], "Draw expected", "#a855f7"

                g1, g2 = st.columns([1, 1.4])
                with g1:
                    render_gauge(gv, gl, gc)
                with g2:
                    st.markdown("<div style='height:18px;'></div>",
                                unsafe_allow_html=True)
                    render_probability_bars(r["p_white"], r["p_draw"], r["p_black"])

                st.markdown(
                    f"<div style='text-align:center;color:#7d8fa8;"
                    f"font-size:0.82rem;margin-top:10px;'>"
                    f"Source: <b style='color:#e6edf3;'>{r['source']}</b>"
                    f"</div>",
                    unsafe_allow_html=True)

                if r["w_if_decisive"] is not None:
                    st.markdown("<div style='height:10px;'></div>",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background:rgba(30,42,61,0.5);"
                        f"border-radius:8px;padding:10px 14px;"
                        f"font-size:0.85rem;color:#c9d1d9;'>"
                        f"<b style='color:#7d8fa8;text-transform:uppercase;"
                        f"font-size:0.72rem;letter-spacing:1px;'>If decisive</b><br>"
                        f"White <b style='color:#22c55e;'>"
                        f"{r['w_if_decisive']:.1%}</b> · "
                        f"Black <b style='color:#3b82f6;'>"
                        f"{r['b_if_decisive']:.1%}</b></div>",
                        unsafe_allow_html=True)

        # ---- Card 2: Decisive model ----
        with c2:
            with st.container(border=True):
                r = pm_results["balanced"]
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:10px;"
                    "margin-bottom:4px;'>"
                    "<span style='font-size:1.4rem;'>⚔️</span>"
                    "<span style='font-size:1.2rem;font-weight:800;"
                    "color:#e6edf3;'>Decisive model</span></div>"
                    "<div style='color:#7d8fa8;font-size:0.82rem;"
                    "margin-bottom:14px;'>balanced · forces a winner</div>",
                    unsafe_allow_html=True)

                if r["p_white"] >= r["p_black"] and r["p_white"] >= r["p_draw"]:
                    gv, gl, gc = r["p_white"], "White wins", "#22c55e"
                elif r["p_black"] >= r["p_white"] and r["p_black"] >= r["p_draw"]:
                    gv, gl, gc = r["p_black"], "Black wins", "#3b82f6"
                else:
                    gv, gl, gc = r["p_draw"], "Draw expected", "#a855f7"

                g1, g2 = st.columns([1, 1.4])
                with g1:
                    render_gauge(gv, gl, gc)
                with g2:
                    st.markdown("<div style='height:18px;'></div>",
                                unsafe_allow_html=True)
                    render_probability_bars(r["p_white"], r["p_draw"], r["p_black"])

                st.markdown(
                    f"<div style='text-align:center;color:#7d8fa8;"
                    f"font-size:0.82rem;margin-top:10px;'>"
                    f"Source: <b style='color:#e6edf3;'>{r['source']}</b>"
                    f"</div>",
                    unsafe_allow_html=True)

                if r["w_if_decisive"] is not None:
                    st.markdown("<div style='height:10px;'></div>",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background:rgba(30,42,61,0.5);"
                        f"border-radius:8px;padding:10px 14px;"
                        f"font-size:0.85rem;color:#c9d1d9;'>"
                        f"<b style='color:#7d8fa8;text-transform:uppercase;"
                        f"font-size:0.72rem;letter-spacing:1px;'>If decisive</b><br>"
                        f"White <b style='color:#22c55e;'>"
                        f"{r['w_if_decisive']:.1%}</b> · "
                        f"Black <b style='color:#3b82f6;'>"
                        f"{r['b_if_decisive']:.1%}</b></div>",
                        unsafe_allow_html=True)


# ============ TAB 5 — Review ============
with tab_review:
    # ---- Header ----
    st.markdown(
        "<div style='display:flex;align-items:center;gap:16px;margin-bottom:18px;'>"
        "<div style='width:52px;height:52px;border-radius:14px;"
        "background:linear-gradient(135deg,#22d3ee,#22c55e);"
        "display:flex;align-items:center;justify-content:center;"
        "font-size:1.7rem;"
        "box-shadow:0 0 22px rgba(34,197,94,0.45);'>"
        "📊</div>"
        "<div style='font-size:1.35rem;font-weight:700;"
        "color:#e6edf3;letter-spacing:-0.3px;'>"
        "Full game review: "
        "<span style='color:#22d3ee;'>evaluation graph</span>, "
        "<span style='color:#22d3ee;'>accuracy</span>, and "
        "<span style='color:#22d3ee;'>move quality</span> for both players."
        "</div></div>",
        unsafe_allow_html=True)

    if "pending_review_pgn" in st.session_state:
        st.session_state["review_pgn"] = st.session_state.pop("pending_review_pgn")
        st.info("📥 Game loaded from the Interactive Board — click Generate Review below.")

    # ---- Input label ----
    st.markdown(
        "<div style='color:#e6edf3;font-weight:700;font-size:1rem;"
        "margin-bottom:6px;'>Paste a PGN to review:</div>",
        unsafe_allow_html=True)

    review_pgn = st.text_area(
        "Paste a PGN to review:",
        height=180,
        key="review_pgn",
        label_visibility="collapsed",
        placeholder="[Event \"...\"]\n[White \"...\"]\n...\n\n1. e4 e5 2. Nf3 ...")

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    if st.button("📊  Generate Review  →", type="primary", key="review_generate", use_container_width=False):
        if not review_pgn.strip():
            st.warning("Please paste a PGN first.")
        else:
            with st.spinner("Analyzing every move... (~20-40 seconds)"):
                result = compute_eval_graph(review_pgn, max_plies=120, depth=10)
            if "error" in result:
                st.session_state.review_data = {"error": result["error"]}
            else:
                st.session_state.review_data = result
                st.session_state.current_review_ply = 0

    graph_data = st.session_state.get("review_data")

    if graph_data and "error" not in graph_data:
        evals = graph_data["evals"]
        num_moves = graph_data["num_moves"]

        st.success(f"Analyzed {num_moves} moves.")
        st.divider()

        final_eval = evals[-1]
        max_eval = max(evals)
        min_eval = min(evals)
        avg_eval = int(sum(evals) / len(evals)) if evals else 0
        num_plies = len(evals) - 1

        # Best move — pick the most impactful "Best"-rated move
        best_move_label = "—"
        best_move_detail = "Engine's top choice"
        try:
            _cats = graph_data.get("categories", [])
            _sans = graph_data.get("sans", [])
            _cpls = graph_data.get("cpls", [])
            _evals = graph_data.get("evals", [])

            # Find every position where the played move was "Best"
            # (skip the very first few plies — those are book moves)
            candidates = []
            for _i in range(7, len(_cats)):
                if _cats[_i] == "Best" and _i < len(_sans):
                    # Score = how much this move changed the eval
                    # (bigger swing = more impactful)
                    if _i - 1 < len(_evals) and _i < len(_evals):
                        swing = abs(_evals[_i] - _evals[_i - 1])
                    else:
                        swing = 0
                    candidates.append((swing, _i))

            if candidates:
                # Pick the best move with the biggest eval swing
                candidates.sort(key=lambda x: -x[0])
                _best_i = candidates[0][1]
                _num = (_best_i + 1) // 2
                _side = "White" if _best_i % 2 == 1 else "Black"
                best_move_label = f"{_num}. {_sans[_best_i]}"
                best_move_detail = f"{_side}'s best move"
        except Exception:
            pass

        card_style = (
            "background:rgba(10,20,45,0.75);"
            "border:1px solid {border};"
            "border-radius:14px;"
            "padding:16px 18px;"
            "display:flex;align-items:center;gap:14px;"
            "box-shadow:0 0 18px {glow};"
        )

        m1, m2, m3, m4 = st.columns(4, gap="small")

        with m1:
            color = "#22c55e" if final_eval >= 0 else "#ef4444"
            verdict = "White is winning" if final_eval >= 200 else (
                "Black is winning" if final_eval <= -200 else "Roughly even")
            st.markdown(
                f"<div style='{card_style.format(border='rgba(34,197,94,0.4)', glow='rgba(34,197,94,0.15)')}'>"
                f"<div style='width:44px;height:44px;border-radius:50%;"
                f"background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.5);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:1.2rem;'>↗</div>"
                f"<div>"
                f"<div style='color:#7d8fa8;font-size:0.75rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.8px;'>Final evaluation</div>"
                f"<div style='color:{color};font-size:1.4rem;font-weight:800;"
                f"line-height:1.1;margin-top:4px;'>{final_eval:+d} cp</div>"
                f"<div style='color:#7d8fa8;font-size:0.75rem;margin-top:2px;'>"
                f"{verdict}</div>"
                f"</div></div>",
                unsafe_allow_html=True)

        with m2:
            st.markdown(
                f"<div style='{card_style.format(border='rgba(168,85,247,0.4)', glow='rgba(168,85,247,0.15)')}'>"
                f"<div style='width:44px;height:44px;border-radius:50%;"
                f"background:rgba(168,85,247,0.15);border:1px solid rgba(168,85,247,0.5);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:1.2rem;'>🎯</div>"
                f"<div>"
                f"<div style='color:#7d8fa8;font-size:0.75rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.8px;'>Best move</div>"
                f"<div style='color:#e6edf3;font-size:1.4rem;font-weight:800;"
                f"line-height:1.1;margin-top:4px;'>{best_move_label}</div>"
                f"<div style='color:#7d8fa8;font-size:0.75rem;margin-top:2px;'>"
                f"{best_move_detail}</div>"
                f"</div></div>",
                unsafe_allow_html=True)

        with m3:
            avg_color = "#22c55e" if avg_eval >= 0 else "#ef4444"
            st.markdown(
                f"<div style='{card_style.format(border='rgba(59,130,246,0.4)', glow='rgba(59,130,246,0.15)')}'>"
                f"<div style='width:44px;height:44px;border-radius:50%;"
                f"background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.5);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:1.2rem;'>📊</div>"
                f"<div>"
                f"<div style='color:#7d8fa8;font-size:0.75rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.8px;'>Avg. evaluation</div>"
                f"<div style='color:{avg_color};font-size:1.4rem;font-weight:800;"
                f"line-height:1.1;margin-top:4px;'>{avg_eval:+d} cp</div>"
                f"<div style='color:#7d8fa8;font-size:0.75rem;margin-top:2px;'>"
                f"Across the game</div>"
                f"</div></div>",
                unsafe_allow_html=True)

        with m4:
            st.markdown(
                f"<div style='{card_style.format(border='rgba(245,158,11,0.4)', glow='rgba(245,158,11,0.15)')}'>"
                f"<div style='width:44px;height:44px;border-radius:50%;"
                f"background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.5);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:1.2rem;'>⏱</div>"
                f"<div>"
                f"<div style='color:#7d8fa8;font-size:0.75rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.8px;'>Game length</div>"
                f"<div style='color:#e6edf3;font-size:1.4rem;font-weight:800;"
                f"line-height:1.1;margin-top:4px;'>{num_plies} plies</div>"
                f"<div style='color:#7d8fa8;font-size:0.75rem;margin-top:2px;'>"
                f"≈ {num_plies // 2} moves per side</div>"
                f"</div></div>",
                unsafe_allow_html=True)

        st.divider()

        opening_name, opening_eco = detect_opening(review_pgn)
        if opening_name:
            st.markdown(
                f"<div style='display:inline-flex;align-items:center;gap:12px;"
                f"background:rgba(10,20,45,0.7);"
                f"border:1px solid rgba(34,211,238,0.45);"
                f"border-radius:14px;padding:14px 22px;margin-bottom:14px;"
                f"box-shadow:0 0 18px rgba(34,211,238,0.2);'>"
                f"<span style='font-size:1.3rem;'>📖</span>"
                f"<span style='color:#e6edf3;font-weight:700;font-size:1rem;'>"
                f"Opening: {opening_name} (ECO {opening_eco})</span>"
                f"<span style='color:#22d3ee;font-size:1.2rem;margin-left:6px;'>›</span>"
                f"</div>",
                unsafe_allow_html=True)
        st.divider()

                # ---- View mode toggle ----
        st.markdown(
            "<div style='color:#e6edf3;font-weight:700;font-size:1rem;"
            "margin-bottom:8px;margin-top:8px;'>View mode:</div>",
            unsafe_allow_html=True)
            
        with st.container(key="view_mode_box"):
            view_mode = st.radio(
                "View mode",
                ["▦  Tabs", "○  Full scroll"],
                horizontal=True,
                label_visibility="collapsed",
            )

        if view_mode.startswith("▦"):
            subtab_eval, subtab_heat, subtab_quality, subtab_explain, subtab_browse = st.tabs([
                "📈 Evaluation", "🔥 Heatmaps", "🎯 Move Quality", "📝 Explanations", "🔍 Browser"])
        else:
            from contextlib import nullcontext
            subtab_eval = subtab_heat = subtab_quality = subtab_explain = subtab_browse = nullcontext()
            st.caption("📜 Full scroll mode — all sections below, scroll to explore.")
            st.divider()

        # ---- Sub-tab 1: Evaluation ----
        with subtab_eval:
            with st.container(border=True):
                # Header row
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:14px;'>"
                    "<div style='width:44px;height:44px;border-radius:12px;"
                    "background:linear-gradient(135deg,#2563eb,#7c3aed);"
                    "display:flex;align-items:center;justify-content:center;"
                    "font-size:1.3rem;"
                    "box-shadow:0 0 16px rgba(59,130,246,0.5);'>📈</div>"
                    "<div>"
                    "<div style='font-size:1.3rem;font-weight:800;"
                    "color:#e6edf3;letter-spacing:-0.3px;'>"
                    "Evaluation Graph</div>"
                    "<div style='color:#7d8fa8;font-size:0.88rem;"
                    "margin-top:2px;'>Position evaluation — White vs Black</div>"
                    "</div></div>",
                    unsafe_allow_html=True)


                # Chart
                x = list(range(len(evals)))
                y = [max(-800, min(800, v)) for v in evals]
                white_adv = [max(0, v) for v in y]
                black_adv = [min(0, v) for v in y]

                fig = go.Figure()

                # --- White advantage ---
                fig.add_trace(go.Scatter(
                    x=x, y=white_adv, mode="lines+markers",
                    line=dict(color="#00e5ff", width=3),
                    marker=dict(size=6, color="#00e5ff", line=dict(width=1, color="#fff")),
                    fill="tozeroy",
                    fillcolor="rgba(0, 229, 255, 0.15)",
                    name="White advantage",
                    hovertemplate="Ply %{x}<br>White +%{y} cp<extra></extra>",
                ))

                # --- Black advantage ---
                fig.add_trace(go.Scatter(
                    x=x, y=black_adv, mode="lines+markers",
                    line=dict(color="#ff2a2a", width=3),
                    marker=dict(size=6, color="#ff2a2a", line=dict(width=1, color="#fff")),
                    fill="tozeroy",
                    fillcolor="rgba(255, 42, 42, 0.15)",
                    name="Black advantage",
                    hovertemplate="Ply %{x}<br>Black %{y} cp<extra></extra>",
                ))

                # --- Endpoint markers with labels ---
                last_x = x[-1] if x else 0
                last_white = white_adv[-1] if white_adv else 0
                last_black = black_adv[-1] if black_adv else 0

                if last_white != 0:
                    fig.add_trace(go.Scatter(
                        x=[last_x], y=[last_white], mode="markers",
                        marker=dict(color="#3b82f6", size=12,
                                    line=dict(color="#ffffff", width=2)),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig.add_annotation(
                        x=last_x, y=last_white,
                        text=f"+{int(last_white)}",
                        showarrow=False, xanchor="left", xshift=14,
                        font=dict(color="#ffffff", size=11, family="Arial Black"),
                        bgcolor="rgba(59,130,246,0.9)",
                        bordercolor="#3b82f6", borderwidth=1,
                        borderpad=4,
                    )

                if last_black != 0:
                    fig.add_trace(go.Scatter(
                        x=[last_x], y=[last_black], mode="markers",
                        marker=dict(color="#ef4444", size=12,
                                    line=dict(color="#ffffff", width=2)),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig.add_annotation(
                        x=last_x, y=last_black,
                        text=f"{int(last_black)}",
                        showarrow=False, xanchor="left", xshift=14,
                        font=dict(color="#ffffff", size=11, family="Arial Black"),
                        bgcolor="rgba(239,68,68,0.9)",
                        bordercolor="#ef4444", borderwidth=1,
                        borderpad=4,
                    )

                fig.add_hline(y=0, line_dash="dash",
                              line_color="rgba(125,143,168,0.5)", line_width=1)

                fig.update_layout(
                    height=420, showlegend=True,                    
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="right",
                            x=0.0, y=-0.25,
                            xanchor="left", yanchor="top",
                            showactive=True,
                            buttons=[
                                dict(label="Show Dots", method="restyle", args=["mode", "lines+markers"]),
                                dict(label="Hide Dots", method="restyle", args=["mode", "lines"]),
                            ],
                            bgcolor="rgba(10,20,45,0.75)",
                            bordercolor="rgba(96,165,250,0.35)",
                            font=dict(color="#c9d1d9", size=11),
                        )
                    ],
                    hoverlabel=dict(
                        bgcolor="rgba(10, 20, 45, 0.9)",
                        bordercolor="rgba(96, 165, 250, 0.5)",
                        font=dict(color="#e6edf3", size=12, family="system-ui, sans-serif")
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top", y=-0.22,
                        xanchor="center", x=0.5,
                        font=dict(color="#c9d1d9", size=11),
                        bgcolor="rgba(10,20,45,0.75)",
                        bordercolor="rgba(96,165,250,0.35)",
                        borderwidth=1,
                    ),
                    xaxis=dict(
                        title=dict(text="Ply (half-move)", standoff=10),
                        gridcolor="rgba(125,143,168,0.08)",
                        color="#7d8fa8",
                        zeroline=False,
                        showline=False,
                    ),
                    yaxis=dict(
                        title="Eval (cp, White POV)",
                        range=[-850, 850],
                        gridcolor="rgba(125,143,168,0.08)",
                        color="#7d8fa8",
                        zeroline=False,
                        showline=False,
                        tickvals=[-500, 0, 500],
                    ),
                    margin=dict(l=50, r=80, t=40, b=90),
                    plot_bgcolor="rgba(5, 15, 35, 0.35)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9d1d9", family="system-ui, sans-serif"),
                )
                import plotly.io as pio

                _modebar_style = """
                <style>
                .modebar-container {
                    top: 0 !important;
                    right: 8px !important;
                }
                .modebar {
                    background: rgba(10,20,45,0.92) !important;
                    border: 1px solid rgba(96,165,250,0.4) !important;
                    border-radius: 10px !important;
                    padding: 4px 6px !important;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
                }
                .modebar-btn path { fill: #c9d1d9 !important; }
                .modebar-btn:hover path { fill: #22c55e !important; }
                .modebar-group { background: transparent !important; }

                /* Custom fullscreen button */
                .fs-btn {
                    display: inline-flex !important;
                    align-items: center;
                    justify-content: center;
                    width: 30px;
                    height: 30px;
                    margin: 0 2px;
                    cursor: pointer;
                    border-radius: 4px;
                }
                .fs-btn:hover { background: rgba(34,197,94,0.15); }
                .fs-btn svg { width: 18px; height: 18px; fill: #c9d1d9; }
                .fs-btn:hover svg { fill: #22c55e; }
                </style>
                """

                _post = """
                setTimeout(function() {
                    var plot = document.querySelector('.js-plotly-plot');
                    if (!plot) return;
                    var traces = plot.querySelectorAll('.scatterlayer .trace');
                    if (traces.length >= 1) {
                        var whiteLines = traces[0].querySelectorAll('.js-line');
                        whiteLines.forEach(function(l) { l.style.filter = 'drop-shadow(0 0 8px rgba(0, 229, 255, 0.8))'; });
                    }
                    if (traces.length >= 2) {
                        var blackLines = traces[1].querySelectorAll('.js-line');
                        blackLines.forEach(function(l) { l.style.filter = 'drop-shadow(0 0 8px rgba(255, 42, 42, 0.8))'; });
                    }

                    var bar = document.querySelector('.modebar');
                    if (!bar || bar.querySelector('.fs-btn')) return;
                    var btn = document.createElement('a');
                    btn.className = 'modebar-btn fs-btn';
                    btn.title = 'Fullscreen';
                    btn.innerHTML =
                        '<svg viewBox="0 0 24 24">' +
                        '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5z' +
                        'm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
                    btn.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        var gd = document.querySelector('.js-plotly-plot');
                        if (!document.fullscreenElement) {
                            if (gd.requestFullscreen) gd.requestFullscreen();
                            else if (gd.webkitRequestFullscreen) gd.webkitRequestFullscreen();
                            else if (gd.msRequestFullscreen) gd.msRequestFullscreen();
                        } else {
                            if (document.exitFullscreen) document.exitFullscreen();
                        }
                    });
                    bar.appendChild(btn);
                }, 500);
                """

                _eval_html = _modebar_style + pio.to_html(
                    fig,
                    include_plotlyjs='cdn',
                    full_html=False,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'responsive': True,
                        'modeBarButtonsToRemove': [
                            'lasso2d', 'select2d', 'toggleSpikelines',
                        ],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'chess_eval_graph',
                            'scale': 2,
                        },
                    },
                    post_script=_post,
                )
                components.html(_eval_html, height=540, scrolling=False)

        # ---- Sub-tab 2: Heatmaps ----
        with subtab_heat:
            # ---- Material Balance ----
            with st.container(border=True):
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:14px;"
                    "margin-bottom:6px;'>"
                    "<div style='width:44px;height:44px;border-radius:12px;"
                    "background:linear-gradient(135deg,#f59e0b,#ef4444);"
                    "display:flex;align-items:center;justify-content:center;"
                    "font-size:1.3rem;"
                    "box-shadow:0 0 16px rgba(245,158,11,0.5);'>⚖️</div>"
                    "<div>"
                    "<div style='font-size:1.3rem;font-weight:800;"
                    "color:#e6edf3;letter-spacing:-0.3px;'>"
                    "Material Balance</div>"
                    "<div style='color:#7d8fa8;font-size:0.88rem;"
                    "margin-top:2px;'>Material balance over the game</div>"
                    "</div></div>",
                    unsafe_allow_html=True)

                # Toggle row
                _t1, _t2, _tspacer = st.columns([1, 1, 3])
                with _t1:
                    show_white = st.toggle("🔵 White material",
                                            value=True, key="mb_white")
                with _t2:
                    show_black = st.toggle("🔴 Black material",
                                            value=True, key="mb_black")

                fens = graph_data.get("fens", [])
                material = [material_balance_from_fen(f) for f in fens]
                x_mat = list(range(len(material)))
                white_mat = [max(0, m) for m in material]
                black_mat = [min(0, m) for m in material]

                fig_mat = go.Figure()

                if show_white:
                    fig_mat.add_trace(go.Scatter(
                        x=x_mat, y=white_mat, mode="lines+markers",
                        line=dict(color="#00e5ff", width=3),
                        marker=dict(size=6, color="#00e5ff", line=dict(width=1, color="#fff")),
                        fill="tozeroy",
                        fillcolor="rgba(0, 229, 255, 0.15)",
                        name="White material",
                        hovertemplate="Ply %{x}<br>White +%{y} pawns<extra></extra>",
                    ))

                if show_black:
                    fig_mat.add_trace(go.Scatter(
                        x=x_mat, y=black_mat, mode="lines+markers",
                        line=dict(color="#ff2a2a", width=3),
                        marker=dict(size=6, color="#ff2a2a", line=dict(width=1, color="#fff")),
                        fill="tozeroy",
                        fillcolor="rgba(255, 42, 42, 0.15)",
                        name="Black material",
                        hovertemplate="Ply %{x}<br>Black %{y} pawns<extra></extra>",
                    ))

                # Endpoint labels
                last_x = x_mat[-1] if x_mat else 0
                last_white = white_mat[-1] if white_mat else 0
                last_black = black_mat[-1] if black_mat else 0

                if show_white and last_white != 0:
                    fig_mat.add_annotation(
                        x=last_x, y=last_white,
                        text=f"+{int(last_white)}",
                        showarrow=False, xanchor="left", xshift=14,
                        font=dict(color="#ffffff", size=11,
                                  family="Arial Black"),
                        bgcolor="rgba(59,130,246,0.9)",
                        bordercolor="#3b82f6", borderwidth=1,
                        borderpad=4,
                    )

                if show_black and last_black != 0:
                    fig_mat.add_annotation(
                        x=last_x, y=last_black,
                        text=f"{int(last_black)}",
                        showarrow=False, xanchor="left", xshift=14,
                        font=dict(color="#ffffff", size=11,
                                  family="Arial Black"),
                        bgcolor="rgba(239,68,68,0.9)",
                        bordercolor="#ef4444", borderwidth=1,
                        borderpad=4,
                    )

                fig_mat.add_hline(y=0, line_dash="dash",
                                  line_color="rgba(125,143,168,0.5)",
                                  line_width=1)

                fig_mat.update_layout(
                    height=440, showlegend=False,                    
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="right",
                            x=0.0, y=-0.22,
                            xanchor="left", yanchor="top",
                            showactive=True,
                            buttons=[
                                dict(label="Show Dots", method="restyle", args=["mode", "lines+markers"]),
                                dict(label="Hide Dots", method="restyle", args=["mode", "lines"]),
                            ],
                            bgcolor="rgba(10,20,45,0.75)",
                            bordercolor="rgba(96,165,250,0.35)",
                            font=dict(color="#c9d1d9", size=11),
                        )
                    ],
                    xaxis=dict(
                        title=dict(text="Ply (half-move)", standoff=15),
                        gridcolor="rgba(125,143,168,0.08)",
                        color="#7d8fa8",
                        zeroline=False,
                        showline=False,
                    ),
                    yaxis=dict(
                        title="Material advantage (pawns)",
                        gridcolor="rgba(125,143,168,0.08)",
                        color="#7d8fa8",
                        zeroline=False,
                        showline=False,
                    ),
                    margin=dict(l=50, r=30, t=20, b=60),
                    plot_bgcolor="rgba(5, 15, 35, 0.35)",
                    paper_bgcolor="rgba(0, 0, 0, 0)",
                    font=dict(color="#c9d1d9",
                              family="system-ui, sans-serif"),
                )

                import plotly.io as pio

                _modebar_style = """
                <style>
                .modebar-container { top: -14px !important; right: 8px !important; }
                .modebar {
                    background: rgba(10,20,45,0.92) !important;
                    border: 1px solid rgba(96,165,250,0.4) !important;
                    border-radius: 10px !important;
                    padding: 4px 6px !important;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
                }
                .modebar-btn path { fill: #c9d1d9 !important; }
                .modebar-btn:hover path { fill: #22c55e !important; }
                .modebar-group { background: transparent !important; }
                .fs-btn {
                    display: inline-flex !important;
                    align-items: center;
                    justify-content: center;
                    width: 30px; height: 30px;
                    margin: 0 2px;
                    cursor: pointer;
                    border-radius: 4px;
                }
                .fs-btn:hover { background: rgba(34,197,94,0.15); }
                .fs-btn svg { width: 18px; height: 18px; fill: #c9d1d9; }
                .fs-btn:hover svg { fill: #22c55e; }
                </style>
                """

                _post = """
                setTimeout(function() {
                    var plot = document.querySelector('.js-plotly-plot');
                    if (!plot) return;
                    var traces = plot.querySelectorAll('.scatterlayer .trace');
                    
                    if (traces.length >= 1) {
                        var whiteLines = traces[0].querySelectorAll('.js-line');
                        whiteLines.forEach(function(l) { l.style.filter = 'drop-shadow(0 0 8px rgba(0, 229, 255, 0.8))'; });
                    }
                    if (traces.length >= 2) {
                        var blackLines = traces[1].querySelectorAll('.js-line');
                        blackLines.forEach(function(l) { l.style.filter = 'drop-shadow(0 0 8px rgba(255, 42, 42, 0.8))'; });
                    }

                    var bar = document.querySelector('.modebar');
                    if (!bar || bar.querySelector('.fs-btn')) return;
                    var btn = document.createElement('a');
                    btn.className = 'modebar-btn fs-btn';
                    btn.title = 'Fullscreen';
                    btn.innerHTML =
                        '<svg viewBox="0 0 24 24">' +
                        '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5z' +
                        'm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
                    btn.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        var gd = document.querySelector('.js-plotly-plot');
                        if (!document.fullscreenElement) {
                            if (gd.requestFullscreen) gd.requestFullscreen();
                            else if (gd.webkitRequestFullscreen) gd.webkitRequestFullscreen();
                            else if (gd.msRequestFullscreen) gd.msRequestFullscreen();
                        } else {
                            if (document.exitFullscreen) document.exitFullscreen();
                        }
                    });
                    bar.appendChild(btn);
                }, 500);
                """

                _mat_html = _modebar_style + pio.to_html(
                    fig_mat,
                    include_plotlyjs='cdn',
                    full_html=False,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'responsive': True,
                        'modeBarButtonsToRemove': [
                            'lasso2d', 'select2d', 'toggleSpikelines',
                        ],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'chess_material_balance',
                            'scale': 2,
                        },
                    },
                    post_script=_post,
                )
                components.html(_mat_html, height=470, scrolling=False)

            st.divider()


            # ---- Piece Activity Heatmap ----
            st.markdown(
                "<div style='display:flex;align-items:center;gap:14px;"
                "margin-top:16px;margin-bottom:16px;'>"
                "<div style='width:44px;height:44px;border-radius:12px;"
                "background:linear-gradient(135deg,#ef4444,#f97316);"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(239,68,68,0.5);'>🔥</div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "Piece <span style='background:linear-gradient(90deg,#f97316 0%,"
                "#ef4444 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "Activity Heatmap</span></div></div>",
                unsafe_allow_html=True)

            _fens_act = graph_data.get("fens", [])
            if _fens_act:
                final_fen = _fens_act[-1]
                grid = piece_activity_heatmap(final_fen)
                
                row_labels = ["8", "7", "6", "5", "4", "3", "2", "1"]
                col_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

                activity_scale = [
                    [0.0, "rgba(200, 85, 25, 0.85)"],
                    [0.25, "rgba(180, 40, 40, 0.5)"],
                    [0.5, "rgba(15, 20, 35, 0.15)"],
                    [0.75, "rgba(45, 100, 200, 0.5)"],
                    [1.0, "rgba(120, 170, 230, 0.85)"],
                ]

                with st.container(border=True):
                    st.markdown(
                        "<div style='display:flex;align-items:center;"
                        "gap:10px;margin-bottom:6px;'>"
                        "<span style='font-size:1.4rem;"
                        "filter: drop-shadow(0 0 8px rgba(96,165,250,0.8));'>"
                        "🔥</span>"
                        "<span style='color:#e6edf3;font-weight:700;"
                        "font-size:1.05rem;'>"
                        "Who controls each square (final position)</span>"
                        "</div>",
                        unsafe_allow_html=True)

                    _max_abs = max(
                        max(abs(v) for row in grid for v in row),
                        1
                    )

                    fig_act = go.Figure(data=go.Heatmap(
                        z=grid, x=col_labels, y=row_labels,
                        colorscale=activity_scale,
                        zmin=-_max_abs, zmax=_max_abs,
                        xgap=2, ygap=2,
                        hovertemplate="Square %{x}%{y}<br>"
                                      "Control: %{z:+d}<extra></extra>",
                        showscale=True,
                        colorbar=dict(
                            thickness=12, len=0.85,
                            tickfont=dict(color="#c9d1d9", size=10),
                            outlinewidth=0,
                        ),
                    ))
                    
                    fig_act.update_layout(
                        height=440,
                        yaxis=dict(
                            autorange="reversed",
                            tickfont=dict(color="#c9d1d9", size=10),
                            showgrid=False,
                            constrain="domain",   # <--- ADDED
                            scaleanchor=None,     # <--- ADDED
                        ),
                        xaxis=dict(
                            side="top",
                            tickfont=dict(color="#c9d1d9", size=10),
                            showgrid=False,
                            constrain="domain",   # <--- ADDED
                        ),
                        margin=dict(l=30, r=60, t=40, b=30),
                        plot_bgcolor="rgba(5, 15, 35, 0.2)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9d1d9",
                                  family="system-ui, sans-serif"),
                    )

                    import plotly.io as pio
                    
                    _act_glow = """
                    function forceResize() {
                        var gd = document.querySelector('.js-plotly-plot');
                        if (gd && window.Plotly) {
                            window.Plotly.relayout(gd, {'autosize': true, 'width': null, 'height': null});
                            window.Plotly.Plots.resize(gd);
                        }
                    }
                    setTimeout(forceResize, 100);
                    setTimeout(forceResize, 500);
                    setTimeout(forceResize, 1500);

                    setTimeout(function() {
                        var plot = document.querySelector('.js-plotly-plot');
                        if (!plot) return;
                        var hm = plot.querySelector('.heatmaplayer');
                        if (!hm) return;
                        if (hm.parentElement.querySelector('.glow-layer')) return;
                        var glow = hm.cloneNode(true);
                        glow.classList.add('glow-layer');
                        glow.style.filter = 'blur(8px) saturate(1.3)';
                        glow.style.opacity = '0.45';
                        glow.style.pointerEvents = 'none';
                        glow.style.mixBlendMode = 'screen';
                        hm.parentElement.insertBefore(glow, hm);
                    }, 800);
                    """

                    _act_html = pio.to_html(
                        fig_act,
                        include_plotlyjs='cdn',
                        full_html=False,
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'responsive': True,
                            'modeBarButtonsToRemove': [
                                'lasso2d', 'select2d', 'toggleSpikelines',
                            ],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'piece_activity',
                                'scale': 2,
                            },
                        },
                        post_script=_act_glow,
                    )
                    components.html(_act_html, height=460, scrolling=False)

                    st.markdown(
                        "<div style='color:#7d8fa8;font-size:0.82rem;"
                        "margin-top:8px;'>🔵 Blue = White controls · "
                        "🔴 Red = Black controls · Dark = balanced.</div>",
                        unsafe_allow_html=True)

            st.divider()

                       
            # Piece Square Heatmap
            st.markdown(
                "<div style='display:flex;align-items:center;gap:14px;"
                "margin-top:16px;margin-bottom:16px;'>"
                "<div style='width:44px;height:44px;border-radius:12px;"
                "background:linear-gradient(135deg,#3b82f6,#a855f7);"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(59,130,246,0.5);'>♟️</div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "Piece <span style='background:linear-gradient(90deg,#3b82f6 0%,"
                "#a855f7 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "Square Heatmap</span></div></div>",
                unsafe_allow_html=True)

            piece_choice = st.selectbox(
                "🎯 Select piece type to visualize:",
                ["All pieces", "Knights", "Bishops", "Rooks",
                 "Queens", "Pawns", "Kings"],
                key="piece_heatmap_choice")

            piece_map = {
                "All pieces": None, "Knights": chess.KNIGHT,
                "Bishops": chess.BISHOP, "Rooks": chess.ROOK,
                "Queens": chess.QUEEN, "Pawns": chess.PAWN,
                "Kings": chess.KING,
            }

            w_grid, b_grid = piece_square_heatmap(
                review_pgn, piece_type=piece_map[piece_choice])

            if w_grid and b_grid:
                row_labels = ["8", "7", "6", "5", "4", "3", "2", "1"]
                col_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

                white_scale = [
                    [0.0, "#0a1428"], [0.2, "#0d2544"],
                    [0.5, "#1e40af"], [0.8, "#3b82f6"], [1.0, "#60a5fa"],
                ]
                black_scale = [
                    [0.0, "#0a1428"], [0.2, "#4c1d0f"],
                    [0.5, "#b91c1c"], [0.8, "#ef4444"], [1.0, "#fb923c"],
                ]

                import plotly.io as pio

                _glow_post = """
                function forceResize() {
                    var gd = document.querySelector('.js-plotly-plot');
                    if (gd && window.Plotly) {
                        window.Plotly.relayout(gd, {'autosize': true, 'width': null, 'height': null});
                        window.Plotly.Plots.resize(gd);
                    }
                }
                setTimeout(forceResize, 100);
                setTimeout(forceResize, 500);
                setTimeout(forceResize, 1500);

                setTimeout(function() {
                    var plot = document.querySelector('.js-plotly-plot');
                    if (!plot) return;
                    var hm = plot.querySelector('.heatmaplayer');
                    if (!hm) return;
                    if (hm.parentElement.querySelector('.glow-layer')) return;
                    var glow = hm.cloneNode(true);
                    glow.classList.add('glow-layer');
                    glow.style.filter = 'blur(8px) saturate(1.3)';
                    glow.style.opacity = '0.45';
                    glow.style.pointerEvents = 'none';
                    glow.style.mixBlendMode = 'screen';
                    hm.parentElement.insertBefore(glow, hm);
                }, 800);
                """

                col_w, col_b = st.columns(2, gap="medium")

                # ---- WHITE HEATMAP ----
                with col_w:
                    with st.container(border=True):
                        st.markdown(
                            "<div style='display:flex;align-items:center;"
                            "gap:10px;margin-bottom:6px;'>"
                            "<span style='font-size:1.4rem;"
                            "filter: drop-shadow(0 0 8px rgba(96,165,250,0.8));'>"
                            "♙</span>"
                            "<span style='color:#e6edf3;font-weight:700;"
                            "font-size:1.05rem;'>White's pieces</span>"
                            "</div>",
                            unsafe_allow_html=True)

                        fig_w = go.Figure(data=go.Heatmap(
                            z=w_grid, x=col_labels, y=row_labels,
                            colorscale=white_scale,
                            zmin=0, zmax=2,
                            xgap=2, ygap=2,
                            hovertemplate="Square %{x}%{y}<br>"
                                          "White visits: %{z}<extra></extra>",
                            showscale=True,
                            colorbar=dict(
                                thickness=12, len=0.85,
                                x=1.02, xanchor="left",
                                tickfont=dict(color="#c9d1d9", size=10),
                                outlinewidth=0,
                            ),
                        ))
                        fig_w.update_layout(
                            height=380,
                            yaxis=dict(
                                autorange="reversed",
                                tickmode="array",
                                tickvals=list(range(8)),
                                ticktext=["8", "7", "6", "5", "4", "3", "2", "1"],
                                tickfont=dict(color="#c9d1d9", size=10),
                                showgrid=True,
                                gridcolor="rgba(125,143,168,0.18)",
                                gridwidth=1,
                                automargin=True,
                                constrain="domain",
                                scaleanchor=None,
                            ),
                            xaxis=dict(
                                side="top",
                                tickmode="array",
                                tickvals=list(range(8)),
                                ticktext=["a", "b", "c", "d", "e", "f", "g", "h"],
                                tickfont=dict(color="#c9d1d9", size=10),
                                showgrid=True,
                                gridcolor="rgba(125,143,168,0.18)",
                                gridwidth=1,
                                automargin=True,
                                constrain="domain",
                            ),
                            margin=dict(l=40, r=80, t=40, b=30),
                            plot_bgcolor="rgba(5, 15, 35, 0.35)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c9d1d9",
                                      family="system-ui, sans-serif"),
                        )

                        _w_html = pio.to_html(
                            fig_w, include_plotlyjs='cdn',
                            full_html=False,
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'responsive': True,
                                'modeBarButtonsToRemove': [
                                    'lasso2d', 'select2d', 'toggleSpikelines',
                                ],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'white_piece_squares',
                                    'scale': 2,
                                },
                            },
                            post_script=_glow_post,
                        )
                        components.html(_w_html, height=400, scrolling=False)

                # ---- BLACK HEATMAP ----
                with col_b:
                    with st.container(border=True):
                        st.markdown(
                            "<div style='display:flex;align-items:center;"
                            "gap:10px;margin-bottom:6px;'>"
                            "<span style='font-size:1.4rem;"
                            "filter: drop-shadow(0 0 8px rgba(249,115,22,0.8));'>"
                            "♟</span>"
                            "<span style='color:#e6edf3;font-weight:700;"
                            "font-size:1.05rem;'>Black's pieces</span>"
                            "</div>",
                            unsafe_allow_html=True)

                        fig_b = go.Figure(data=go.Heatmap(
                            z=b_grid, x=col_labels, y=row_labels,
                            colorscale=black_scale,
                            zmin=0, zmax=2,
                            xgap=2, ygap=2,
                            hovertemplate="Square %{x}%{y}<br>"
                                          "Black visits: %{z}<extra></extra>",
                            showscale=True,
                            colorbar=dict(
                                thickness=12, len=0.85,
                                x=1.02, xanchor="left",
                                tickfont=dict(color="#c9d1d9", size=10),
                                outlinewidth=0,
                            ),
                        ))
                        fig_b.update_layout(
                            height=380,
                            yaxis=dict(
                                autorange="reversed",
                                tickmode="array",
                                tickvals=list(range(8)),
                                ticktext=["8", "7", "6", "5", "4", "3", "2", "1"],
                                tickfont=dict(color="#c9d1d9", size=10),
                                showgrid=True,
                                gridcolor="rgba(125,143,168,0.18)",
                                gridwidth=1,
                                automargin=True,
                                constrain="domain",
                                scaleanchor=None,
                            ),
                            xaxis=dict(
                                side="top",
                                tickmode="array",
                                tickvals=list(range(8)),
                                ticktext=["a", "b", "c", "d", "e", "f", "g", "h"],
                                tickfont=dict(color="#c9d1d9", size=10),
                                showgrid=True,
                                gridcolor="rgba(125,143,168,0.18)",
                                gridwidth=1,
                                automargin=True,
                                constrain="domain",
                            ),
                            margin=dict(l=40, r=80, t=40, b=30),
                            plot_bgcolor="rgba(5, 15, 35, 0.35)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c9d1d9",
                                      family="system-ui, sans-serif"),
                        )

                        _b_html = pio.to_html(
                            fig_b, include_plotlyjs='cdn',
                            full_html=False,
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'responsive': True,
                                'modeBarButtonsToRemove': [
                                    'lasso2d', 'select2d', 'toggleSpikelines',
                                ],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'black_piece_squares',
                                    'scale': 2,
                                },
                            },
                            post_script=_glow_post,
                        )
                        components.html(_b_html, height=400, scrolling=False)

                st.markdown(
                    "<div style='color:#7d8fa8;font-size:0.82rem;"
                    "margin-top:8px;'>ℹ️ Brighter square = visited more often.</div>",
                    unsafe_allow_html=True)

            st.divider()



            # King Safety Timeline
            st.markdown(
                "<div style='display:flex;align-items:center;gap:14px;"
                "margin-top:16px;margin-bottom:16px;'>"
                "<div style='width:44px;height:44px;border-radius:12px;"
                "background:linear-gradient(135deg,#f59e0b,#fbbf24);"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(245,158,11,0.5);'>👑</div>"
                "<div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "King <span style='background:linear-gradient(90deg,#3b82f6 0%,"
                "#22d3ee 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "Safety Timeline</span></div>"
                "<div style='color:#7d8fa8;font-size:0.95rem;margin-top:2px;'>"
                "King safety over the game (higher = safer)</div>"
                "</div></div>",
                unsafe_allow_html=True)

            if fens:
                with st.container(border=True):
                    w_safety, b_safety = king_safety_timeline(fens)
                    x_ks = list(range(len(w_safety)))
                    
                    fig_ks = go.Figure()
                    # White King Trace
                    fig_ks.add_trace(go.Scatter(
                        x=x_ks, y=w_safety, mode="lines+markers",
                        line=dict(color="#00e5ff", width=3),
                        marker=dict(size=6, color="#00e5ff", line=dict(width=1, color="#fff")),
                        fill="tozeroy",
                        fillcolor="rgba(0, 229, 255, 0.15)",
                        name="White king",
                        hovertemplate="Ply %{x}<br>White safety: %{y}<extra></extra>"
                    ))
                    # Black King Trace
                    fig_ks.add_trace(go.Scatter(
                        x=x_ks, y=b_safety, mode="lines+markers",
                        line=dict(color="#ff2a2a", width=3),
                        marker=dict(size=6, color="#ff2a2a", line=dict(width=1, color="#fff")),
                        fill="tozeroy",
                        fillcolor="rgba(255, 42, 42, 0.15)",
                        name="Black king",
                        hovertemplate="Ply %{x}<br>Black safety: %{y}<extra></extra>"
                    ))

                    fig_ks.update_layout(
                        height=400, showlegend=True,                        
                        updatemenus=[
                            dict(
                                type="buttons",
                                direction="right",
                                x=0.0, y=-0.22,
                                xanchor="left", yanchor="top",
                                showactive=True,
                                buttons=[
                                    dict(label="Show Dots", method="restyle", args=["mode", "lines+markers"]),
                                    dict(label="Hide Dots", method="restyle", args=["mode", "lines"]),
                                ],
                                bgcolor="rgba(10,20,45,0.75)",
                                bordercolor="rgba(96,165,250,0.35)",
                                font=dict(color="#c9d1d9", size=11),
                            )
                        ],
                        hoverlabel=dict(
                            bgcolor="rgba(10, 20, 45, 0.9)",
                            bordercolor="rgba(96, 165, 250, 0.5)",
                            font=dict(color="#e6edf3", size=12, family="system-ui, sans-serif")
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="right", x=1,
                            font=dict(color="#c9d1d9", size=11),
                            bgcolor="rgba(10,20,45,0.75)",
                            bordercolor="rgba(96,165,250,0.35)",
                            borderwidth=1,
                        ),
                        xaxis=dict(
                            title=dict(text="Ply (half-move)", standoff=15),
                            gridcolor="rgba(125,143,168,0.08)",
                            color="#7d8fa8",
                            zeroline=False, showline=False,
                        ),
                        yaxis=dict(
                            title="Safety score",
                            gridcolor="rgba(125,143,168,0.08)",
                            color="#7d8fa8",
                            zeroline=False, showline=False,
                        ),
                        margin=dict(l=50, r=30, t=60, b=60),
                        plot_bgcolor="rgba(5, 15, 35, 0.35)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9d1d9", family="system-ui, sans-serif"),
                    )

                    import plotly.io as pio
                    
                    _post_ks = """
                    setTimeout(function() {
                        var plot = document.querySelector('.js-plotly-plot');
                        if (!plot) return;

                        // 1. Glow effect
                        var traces = plot.querySelectorAll('.scatterlayer .trace');
                        if (traces.length >= 2) {
                            var whiteLines = traces[0].querySelectorAll('.js-line');
                            whiteLines.forEach(function(l) { l.style.filter = 'drop-shadow(0 0 8px rgba(0, 229, 255, 0.8))'; });
                            
                            var blackLines = traces[1].querySelectorAll('.js-line');
                            blackLines.forEach(function(l) { l.style.filter = 'drop-shadow(0 0 8px rgba(255, 42, 42, 0.8))'; });
                        }

                        // 2. Fullscreen button
                        var bar = document.querySelector('.modebar');
                        if (!bar || bar.querySelector('.fs-btn')) return;
                        var btn = document.createElement('a');
                        btn.className = 'modebar-btn fs-btn';
                        btn.title = 'Fullscreen';
                        btn.innerHTML =
                            '<svg viewBox="0 0 24 24">' +
                            '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5z' +
                            'm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
                        btn.addEventListener('click', function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            var gd = document.querySelector('.js-plotly-plot');
                            if (!document.fullscreenElement) {
                                if (gd.requestFullscreen) gd.requestFullscreen();
                                else if (gd.webkitRequestFullscreen) gd.webkitRequestFullscreen();
                                else if (gd.msRequestFullscreen) gd.msRequestFullscreen();
                            } else {
                                if (document.exitFullscreen) document.exitFullscreen();
                            }
                        });
                        bar.appendChild(btn);
                    }, 500);
                    """

                    _ks_html = _modebar_style + pio.to_html(
                        fig_ks,
                        include_plotlyjs='cdn',
                        full_html=False,
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'responsive': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'toggleSpikelines'],
                            'toImageButtonOptions': {'format': 'png', 'filename': 'king_safety', 'scale': 2},
                        },
                        post_script=_post_ks,
                    )
                    components.html(_ks_html, height=440, scrolling=False)

                st.markdown(
                    "<div style='display:flex;gap:12px;margin-top:12px;'>"
                    "<div style='display:flex;align-items:center;gap:8px;"
                    "background:rgba(10,20,45,0.7);border:1px solid rgba(34,211,238,0.3);"
                    "padding:6px 14px;border-radius:20px;'>"
                    "<span style='width:10px;height:10px;border-radius:50%;"
                    "background:#00e5ff;box-shadow:0 0 8px #00e5ff;'></span>"
                    "<span style='color:#c9d1d9;font-size:0.82rem;font-weight:600;'>"
                    "White king</span></div>"
                    "<div style='display:flex;align-items:center;gap:8px;"
                    "background:rgba(10,20,45,0.7);border:1px solid rgba(239,68,68,0.3);"
                    "padding:6px 14px;border-radius:20px;'>"
                    "<span style='width:10px;height:10px;border-radius:50%;"
                    "background:#ff2a2a;box-shadow:0 0 8px #ff2a2a;'></span>"
                    "<span style='color:#c9d1d9;font-size:0.82rem;font-weight:600;'>"
                    "Black king</span></div>"
                    "</div>",
                    unsafe_allow_html=True)

            st.divider()

        # ---- Sub-tab 3: Move Quality ----
        with subtab_quality:
        # Move Quality Distribution
            st.markdown(
                "<div style='display:flex;align-items:center;gap:14px;"
                "margin-top:16px;margin-bottom:16px;'>"
                "<div style='width:44px;height:44px;border-radius:12px;"
                "background:linear-gradient(135deg,#ef4444,#f97316);"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(239,68,68,0.5);'>🎯</div>"
                "<div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "Move <span style='background:linear-gradient(90deg,#3b82f6 0%,"
                "#22d3ee 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "Quality Distribution</span></div>"
                "<div style='color:#7d8fa8;font-size:0.95rem;margin-top:2px;'>"
                "Analyze the quality of moves and improve your strategy</div>"
                "</div></div>",
                unsafe_allow_html=True)

            categories = graph_data.get("categories", [])
            white_cats = []
            black_cats = []
            for i in range(1, len(categories)):
                if i % 2 == 1:
                    white_cats.append(categories[i])
                else:
                    black_cats.append(categories[i])
                    
            cat_list = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
            emoji_map = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                         "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}
            color_map = {
                "Best": "#facc15", "Excellent": "#22c55e", 
                "Good": "#60a5fa", "Inaccuracy": "#f97316", 
                "Mistake": "#ef4444", "Blunder": "#ec4899"
            }
            
            white_counts = [white_cats.count(c) for c in cat_list]
            black_counts = [black_cats.count(c) for c in cat_list]
            max_count = max(white_counts + black_counts + [1])
            
            col_legend, col_chart = st.columns([1, 3.5], gap="medium")

            with col_legend:
                with st.container(border=True):
                    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                    
                    # Exact custom SVGs matching the reference image
                    svg_icons = {
                        "Blunder": '<svg viewBox="0 0 24 24" width="14" height="14" fill="#ec4899"><path d="M12 2l2.5 6.5L21 11l-6.5 2.5L12 20l-2.5-6.5L3 11l6.5-2.5L12 2z"/></svg>',
                        "Mistake": '<svg viewBox="0 0 24 24" width="14" height="14" fill="#ec4899"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>',
                        "Inaccuracy": '<svg viewBox="0 0 24 24" width="14" height="14" fill="#f97316"><path d="M12 2L1 21h22L12 2zm1 15h-2v-2h2v2zm0-4h-2V9h2v4z"/></svg>',
                        "Good": '<svg viewBox="0 0 24 24" width="14" height="14" fill="#22c55e"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-9 14l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>',
                        "Excellent": '<svg viewBox="0 0 24 24" width="14" height="14"><circle cx="12" cy="12" r="10" fill="#22c55e" opacity="0.2" stroke="#22c55e" stroke-width="2"/><path fill="#facc15" d="M12 6l1.8 3.7 4.1.6-3 2.9.7 4.1-3.6-1.9-3.6 1.9.7-4.1-3-2.9 4.1-.6z"/></svg>',
                        "Best": '<svg viewBox="0 0 24 24" width="14" height="14" fill="#facc15"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/></svg>'
                    }
                    
                    for cat in reversed(cat_list):
                        c_color = color_map[cat]
                        svg = svg_icons.get(cat, "")
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:12px;"
                            f"padding:10px 0;'>"
                            f"<div style='width:28px;height:28px;border-radius:50%;"
                            f"background:rgba(255,255,255,0.05);"
                            f"border:1px solid {c_color}44;"
                            f"display:flex;align-items:center;justify-content:center;"
                            f"box-shadow:0 0 10px {c_color}33;'>{svg}</div>"
                            f"<span style='color:#e6edf3;font-weight:600;"
                            f"font-size:0.95rem;'>{cat}</span></div>",
                            unsafe_allow_html=True
                        )

            with col_chart:
                with st.container(border=True):
                    fig2 = go.Figure()
                    
                    # White bars
                    fig2.add_trace(go.Bar(
                        name="White", y=cat_list, x=white_counts, orientation="h",
                        marker=dict(
                            color="#00e5ff",
                            line=dict(color="#ffffff", width=1)
                        ),
                        hovertemplate="White %{y}: %{x}<extra></extra>"))
                        
                    # Black bars
                    fig2.add_trace(go.Bar(
                        name="Black", y=cat_list, x=[-v for v in black_counts], orientation="h",
                        marker=dict(
                            color="#ff2a2a",
                            line=dict(color="#ffffff", width=1)
                        ),
                        hovertemplate="Black %{y}: %{customdata}<extra></extra>",
                        customdata=black_counts))

                    fig2.update_layout(
                        barmode="overlay", height=400,
                        hoverlabel=dict(
                            bgcolor="rgba(10, 20, 45, 0.9)",
                            bordercolor="rgba(96, 165, 250, 0.5)",
                            font=dict(color="#e6edf3", size=12, family="system-ui, sans-serif")
                        ),
                        xaxis=dict(
                            title="Number of moves",
                            range=[-max_count - 1, max_count + 1],
                            tickvals=[-max_count, 0, max_count],
                            ticktext=[str(max_count), "0", str(max_count)],
                            gridcolor="rgba(125,143,168,0.08)",
                            color="#7d8fa8",
                            zeroline=True,
                            zerolinecolor="rgba(125,143,168,0.3)",
                        ),
                        yaxis=dict(
                            showticklabels=True,
                            showgrid=False,
                            tickmode="array",
                            tickvals=cat_list,
                            ticktext=[f"{emoji_map[c]}  {c}" for c in cat_list],
                            tickfont=dict(color="#e6edf3", size=13, family="system-ui, sans-serif")
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5,
                            font=dict(color="#c9d1d9", size=11),
                            bgcolor="rgba(10,20,45,0.75)",
                            bordercolor="rgba(96,165,250,0.35)",
                            borderwidth=1,
                        ),
                        margin=dict(l=120, r=20, t=60, b=50),
                        plot_bgcolor="rgba(5, 15, 35, 0.35)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9d1d9", family="system-ui, sans-serif"),
                    )

                    import plotly.io as pio
                    
                    _post_mq = """
                    setTimeout(function() {
                        var plot = document.querySelector('.js-plotly-plot');
                        if (!plot) return;
                        var traces = plot.querySelectorAll('.barlayer .trace');
                        if (traces.length >= 1) {
                            var whiteBars = traces[0].querySelectorAll('.point path');
                            whiteBars.forEach(function(b) { b.style.filter = 'drop-shadow(0 0 6px rgba(0, 229, 255, 0.6))'; });
                        }
                        if (traces.length >= 2) {
                            var blackBars = traces[1].querySelectorAll('.point path');
                            blackBars.forEach(function(b) { b.style.filter = 'drop-shadow(0 0 6px rgba(255, 42, 42, 0.6))'; });
                        }

                        var bar = document.querySelector('.modebar');
                        if (!bar || bar.querySelector('.fs-btn')) return;
                        var btn = document.createElement('a');
                        btn.className = 'modebar-btn fs-btn';
                        btn.title = 'Fullscreen';
                        btn.innerHTML =
                            '<svg viewBox="0 0 24 24">' +
                            '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5z' +
                            'm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
                        btn.addEventListener('click', function(e) {
                            e.preventDefault(); e.stopPropagation();
                            var gd = document.querySelector('.js-plotly-plot');
                            if (!document.fullscreenElement) {
                                if (gd.requestFullscreen) gd.requestFullscreen();
                                else if (gd.webkitRequestFullscreen) gd.webkitRequestFullscreen();
                                else if (gd.msRequestFullscreen) gd.msRequestFullscreen();
                            } else {
                                if (document.exitFullscreen) document.exitFullscreen();
                            }
                        });
                        bar.appendChild(btn);
                    }, 500);
                    """

                    _mq_html = _modebar_style + pio.to_html(
                        fig2,
                        include_plotlyjs='cdn',
                        full_html=False,
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'responsive': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'toggleSpikelines'],
                            'toImageButtonOptions': {'format': 'png', 'filename': 'move_quality', 'scale': 2},
                        },
                        post_script=_post_mq,
                    )
                    components.html(_mq_html, height=420, scrolling=False)

            st.divider()

            # Accuracy
            st.markdown(
                "<div style='display:flex;align-items:center;gap:14px;"
                "margin-top:16px;margin-bottom:16px;'>"
                "<div style='width:44px;height:44px;border-radius:50%;"
                "background:rgba(34,211,238,0.1);border:2px solid #22d3ee;"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(34,211,238,0.4);'>🎯</div>"
                "<div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "Accuracy <span style='background:linear-gradient(90deg,#22d3ee 0%,"
                "#3b82f6 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "per Player</span></div>"
                "<div style='color:#7d8fa8;font-size:0.95rem;margin-top:2px;'>"
                "Compare the precision and consistency of each player's moves</div>"
                "</div></div>",
                unsafe_allow_html=True)

            cpls = graph_data.get("cpls", [])
            white_cpls = []
            black_cpls = []
            for i in range(1, len(cpls)):
                if i % 2 == 1:
                    white_cpls.append(cpls[i])
                else:
                    black_cpls.append(cpls[i])
                    
            def compute_accuracy(cpl_list):
                if not cpl_list:
                    return 0.0
                avg_cpl = sum(cpl_list) / len(cpl_list)
                return max(0.0, min(100.0, 100.0 - avg_cpl / 5.0))
                
            white_acc = compute_accuracy(white_cpls)
            black_acc = compute_accuracy(black_cpls)
            white_avg = int(sum(white_cpls)/len(white_cpls)) if white_cpls else 0
            black_avg = int(sum(black_cpls)/len(black_cpls)) if black_cpls else 0

            card_col1, card_col2 = st.columns(2, gap="medium")

            # White Card
            with card_col1:
                st.markdown(
                    f"<div style='background:rgba(10,20,45,0.6);"
                    f"border:1px solid rgba(34,211,238,0.4);border-radius:14px;"
                    f"padding:16px 20px;display:flex;align-items:center;gap:16px;"
                    f"box-shadow:0 0 20px rgba(34,211,238,0.15);'>"
                    f"<div style='width:40px;height:40px;border-radius:50%;"
                    f"background:rgba(34,211,238,0.15);display:flex;align-items:center;"
                    f"justify-content:center;font-size:1.2rem;'>👑</div>"
                    f"<div>"
                    f"<div style='color:#e6edf3;font-weight:700;font-size:1.05rem;'>White accuracy</div>"
                    f"<div style='color:#22d3ee;font-size:1.8rem;font-weight:800;line-height:1.1;margin-top:4px;'>{white_acc:.1f}%</div>"
                    f"<div style='color:#7d8fa8;font-size:0.8rem;margin-top:4px;'>📈 Avg. CPL: {white_avg}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

            # Black Card
            with card_col2:
                st.markdown(
                    f"<div style='background:rgba(45,10,20,0.6);"
                    f"border:1px solid rgba(239,68,68,0.4);border-radius:14px;"
                    f"padding:16px 20px;display:flex;align-items:center;gap:16px;"
                    f"box-shadow:0 0 20px rgba(239,68,68,0.15);'>"
                    f"<div style='width:40px;height:40px;border-radius:50%;"
                    f"background:rgba(239,68,68,0.15);display:flex;align-items:center;"
                    f"justify-content:center;font-size:1.2rem;'>👑</div>"
                    f"<div>"
                    f"<div style='color:#e6edf3;font-weight:700;font-size:1.05rem;'>Black accuracy</div>"
                    f"<div style='color:#ef4444;font-size:1.8rem;font-weight:800;line-height:1.1;margin-top:4px;'>{black_acc:.1f}%</div>"
                    f"<div style='color:#7d8fa8;font-size:0.8rem;margin-top:4px;'>📈 Avg. CPL: {black_avg}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            chart_col1, chart_col2 = st.columns(2, gap="medium")

            import plotly.io as pio
            
            _post_acc = """
            setTimeout(function() {
                var plot = document.querySelector('.js-plotly-plot');
                if (!plot) return;
                var bars = plot.querySelectorAll('.barlayer .point path');
                bars.forEach(function(b) {
                    var fill = b.getAttribute('fill');
                    b.style.filter = 'drop-shadow(0 0 10px ' + fill + ')';
                });
            }, 500);
            """

            # White Chart
            with chart_col1:
                with st.container(border=True):
                    fig_acc_w = go.Figure(data=[go.Bar(
                        x=["White"], y=[white_acc],
                        marker=dict(color="#00e5ff", line=dict(width=0)),
                        width=0.6,
                    )])
                    fig_acc_w.update_layout(
                        height=280,
                        yaxis=dict(range=[0, 105], title="Accuracy %", gridcolor="rgba(125,143,168,0.08)", color="#7d8fa8"),
                        xaxis=dict(tickfont=dict(color="#e6edf3", size=13)),
                        margin=dict(l=40, r=20, t=30, b=30),
                        plot_bgcolor="rgba(5, 15, 35, 0.35)", paper_bgcolor="rgba(0,0,0,0)",
                        annotations=[dict(
                            x=0, y=white_acc, text=f"{white_acc:.1f}%",
                            showarrow=False, yshift=15,
                            bgcolor="rgba(10,20,45,0.9)", bordercolor="#00e5ff",
                            borderwidth=1, borderpad=4,
                            font=dict(color="#00e5ff", size=12, weight="bold")
                        )]
                    )
                    _w_acc_html = pio.to_html(fig_acc_w, include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False}, post_script=_post_acc)
                    components.html(_w_acc_html, height=300, scrolling=False)

            # Black Chart
            with chart_col2:
                with st.container(border=True):
                    fig_acc_b = go.Figure(data=[go.Bar(
                        x=["Black"], y=[black_acc],
                        marker=dict(color="#ff2a2a", line=dict(width=0)),
                        width=0.6,
                    )])
                    fig_acc_b.update_layout(
                        height=280,
                        yaxis=dict(range=[0, 105], title="Accuracy %", gridcolor="rgba(125,143,168,0.08)", color="#7d8fa8"),
                        xaxis=dict(tickfont=dict(color="#e6edf3", size=13)),
                        margin=dict(l=40, r=20, t=30, b=30),
                        plot_bgcolor="rgba(5, 15, 35, 0.35)", paper_bgcolor="rgba(0,0,0,0)",
                        annotations=[dict(
                            x=0, y=black_acc, text=f"{black_acc:.1f}%",
                            showarrow=False, yshift=15,
                            bgcolor="rgba(10,20,45,0.9)", bordercolor="#ff2a2a",
                            borderwidth=1, borderpad=4,
                            font=dict(color="#ff2a2a", size=12, weight="bold")
                        )]
                    )
                    _b_acc_html = pio.to_html(fig_acc_b, include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False}, post_script=_post_acc)
                    components.html(_b_acc_html, height=300, scrolling=False)

            st.divider()

            # CPL Timeline
            st.markdown(
                "<div style='display:flex;align-items:center;gap:14px;"
                "margin-top:16px;margin-bottom:16px;'>"
                "<div style='width:44px;height:44px;border-radius:12px;"
                "background:linear-gradient(135deg,#3b82f6,#ec4899);"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(236,72,153,0.5);'>📊</div>"
                "<div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "CPL <span style='background:linear-gradient(90deg,#3b82f6 0%,"
                "#ec4899 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "Timeline</span></div>"
                "<div style='color:#7d8fa8;font-size:0.95rem;margin-top:2px;'>"
                "Track how your position changes over time</div>"
                "</div></div>",
                unsafe_allow_html=True)

            # Filter buttons
            cpl_filter = st.radio(
                "CPL View",
                ["All", "White only", "Black only"],
                horizontal=True,
                label_visibility="collapsed",
                key="cpl_filter"
            )

            cpls = graph_data.get("cpls", [])
            sans = graph_data.get("sans", [])
            
            if cpls and len(cpls) > 1:
                white_x, white_y = [], []
                black_x, black_y = [], []

                for i, cpl in enumerate(cpls[1:]):
                    ply_index = i + 1
                    if ply_index % 2 == 1:
                        white_x.append(ply_index)
                        white_y.append(cpl)
                    else:
                        black_x.append(ply_index)
                        black_y.append(cpl)

                fig4 = go.Figure()

                                # Add White trace
                if cpl_filter in ["All", "White only"]:
                    fig4.add_trace(go.Bar(
                        x=white_x, y=white_y, name="White",
                        width=0.6,  # <--- ADD THIS LINE
                        marker=dict(
                            color="#00e5ff",
                            line=dict(color="rgba(255,255,255,0.2)", width=1)
                        )
                    ))

                # Add Black trace with diagonal pattern
                if cpl_filter in ["All", "Black only"]:
                    fig4.add_trace(go.Bar(
                        x=black_x, y=black_y, name="Black",
                        width=0.6,  # <--- ADD THIS LINE
                        marker=dict(
                            color="#ff2a2a",
                            line=dict(color="rgba(255,255,255,0.2)", width=1),
                            pattern=dict(
                                shape="/",
                                fgcolor="rgba(255,255,255,0.4)",
                                fillmode="overlay",
                                solidity=0.4
                            )
                        )
                    ))

                with st.container(border=True):
                    fig4.update_layout(
                        barmode="overlay",
                        height=380,
                        hoverlabel=dict(
                            bgcolor="rgba(10, 20, 45, 0.9)",
                            bordercolor="rgba(96, 165, 250, 0.5)",
                            font=dict(color="#e6edf3", size=12, family="system-ui, sans-serif")
                        ),
                        xaxis=dict(
                            title="Ply (half-move)",
                            gridcolor="rgba(125,143,168,0.08)",
                            color="#7d8fa8",
                            zeroline=False,
                        ),
                        yaxis=dict(
                            title="CPL (centipawns lost)",
                            gridcolor="rgba(125,143,168,0.08)",
                            color="#7d8fa8",
                            zeroline=False,
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5,
                            font=dict(color="#c9d1d9", size=11),
                            bgcolor="rgba(10,20,45,0.75)",
                            bordercolor="rgba(96,165,250,0.35)",
                            borderwidth=1,
                        ),
                        margin=dict(l=50, r=20, t=60, b=50),
                        plot_bgcolor="rgba(5, 15, 35, 0.35)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9d1d9", family="system-ui, sans-serif"),
                    )

                    import plotly.io as pio
                    
                    _post_cpl = """
                    setTimeout(function() {
                        var plot = document.querySelector('.js-plotly-plot');
                        if (!plot) return;
                        
                        // 1. Glow effect
                        var bars = plot.querySelectorAll('.barlayer .point path');
                        bars.forEach(function(b) {
                            var fill = b.getAttribute('fill') || '';
                            if (fill.includes('255') || fill.includes('ff2a')) {
                                b.style.filter = 'drop-shadow(0 0 10px rgba(255, 42, 42, 0.8))';
                            } else if (fill.includes('229') || fill.includes('00e5')) {
                                b.style.filter = 'drop-shadow(0 0 10px rgba(0, 229, 255, 0.8))';
                            } else {
                                b.style.filter = 'drop-shadow(0 0 10px rgba(255, 255, 255, 0.4))';
                            }
                        });

                        // 2. Fullscreen button
                        var bar = document.querySelector('.modebar');
                        if (!bar || bar.querySelector('.fs-btn')) return;
                        var btn = document.createElement('a');
                        btn.className = 'modebar-btn fs-btn';
                        btn.title = 'Fullscreen';
                        btn.innerHTML =
                            '<svg viewBox="0 0 24 24">' +
                            '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5z' +
                            'm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
                        btn.addEventListener('click', function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            var gd = document.querySelector('.js-plotly-plot');
                            if (!document.fullscreenElement) {
                                if (gd.requestFullscreen) gd.requestFullscreen();
                                else if (gd.webkitRequestFullscreen) gd.webkitRequestFullscreen();
                                else if (gd.msRequestFullscreen) gd.msRequestFullscreen();
                            } else {
                                if (document.exitFullscreen) document.exitFullscreen();
                            }
                        });
                        bar.appendChild(btn);
                    }, 800);
                    """

                    # Prepend _modebar_style and enable the toolbar in the config
                    _cpl_html = _modebar_style + pio.to_html(
                        fig4,
                        include_plotlyjs='cdn',
                        full_html=False,
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'responsive': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'toggleSpikelines'],
                            'toImageButtonOptions': {'format': 'png', 'filename': 'cpl_timeline', 'scale': 2},
                        },
                        post_script=_post_cpl,
                    )
                    components.html(_cpl_html, height=400, scrolling=False)

            # Bottom Legend Pills
            st.markdown(
                "<div style='display:flex;gap:12px;margin-top:12px;'>"
                "<div style='display:flex;align-items:center;gap:8px;"
                "background:rgba(10,20,45,0.7);border:1px solid rgba(34,211,238,0.3);"
                "padding:6px 14px;border-radius:20px;'>"
                "<span style='width:10px;height:10px;border-radius:50%;"
                "background:#00e5ff;box-shadow:0 0 8px #00e5ff;'></span>"
                "<span style='color:#c9d1d9;font-size:0.82rem;font-weight:600;'>"
                "Blue border = White</span></div>"
                "<div style='display:flex;align-items:center;gap:8px;"
                "background:rgba(10,20,45,0.7);border:1px solid rgba(239,68,68,0.3);"
                "padding:6px 14px;border-radius:20px;'>"
                "<span style='width:10px;height:10px;border-radius:50%;"
                "background:#ff2a2a;box-shadow:0 0 8px #ff2a2a;'></span>"
                "<span style='color:#c9d1d9;font-size:0.82rem;font-weight:600;'>"
                "Red border = Black</span></div>"
                "</div>",
                unsafe_allow_html=True)

            st.divider()

            # Mistake Density by Game Phase
            st.markdown(
                "<div style='display:flex;align-items:center;gap:14px;"
                "margin-top:16px;margin-bottom:16px;'>"
                "<div style='width:44px;height:44px;border-radius:50%;"
                "background:rgba(236,72,153,0.1);border:2px solid #ec4899;"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(236,72,153,0.4);'>🎯</div>"
                "<div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "Mistake <span style='background:linear-gradient(90deg,#3b82f6 0%,"
                "#22d3ee 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "Density by Phase</span></div>"
                "<div style='color:#7d8fa8;font-size:0.95rem;margin-top:2px;'>"
                "Compare the number of mistakes + inaccuracies + blunders across each phase</div>"
                "</div></div>",
                unsafe_allow_html=True)

            categories = graph_data.get("categories", [])
            opening_end = 30
            middlegame_end = 80
            phases = {"Opening (1-15)": [0, 0], "Middlegame (16-40)": [0, 0], "Endgame (40+)": [0, 0]}
            
            for i, cat in enumerate(categories):
                if i == 0: continue
                is_bad = cat in ("Inaccuracy", "Mistake", "Blunder")
                if not is_bad: continue
                if i <= opening_end: phase = "Opening (1-15)"
                elif i <= middlegame_end: phase = "Middlegame (16-40)"
                else: phase = "Endgame (40+)"
                if i % 2 == 1: phases[phase][1] += 1  # White
                else: phases[phase][0] += 1           # Black

            # Add emojis to phase names for the x-axis
            phase_names = ["♟️ Opening (1-15)", "♞ Middlegame (16-40)", "♚ Endgame (40+)"]
            base_phases = ["Opening (1-15)", "Middlegame (16-40)", "Endgame (40+)"]
            
            black_mistakes = [phases[p][0] for p in base_phases]
            white_mistakes = [phases[p][1] for p in base_phases]

            with st.container(border=True):
                fig5 = go.Figure()

                # Black Bars (Red)
                fig5.add_trace(go.Bar(
                    name="Black", 
                    x=phase_names, 
                    y=black_mistakes,
                    marker=dict(
                        color="rgba(255, 42, 42, 0.15)", 
                        line=dict(color="#ff2a2a", width=2)
                    ),
                    text=black_mistakes, 
                    textposition="outside",
                    textfont=dict(color="#e6edf3", size=12, weight="bold"),
                    hovertemplate="Black %{x}: %{y}<extra></extra>"
                ))

                # White Bars (Blue)
                fig5.add_trace(go.Bar(
                    name="White", 
                    x=phase_names, 
                    y=white_mistakes,
                    marker=dict(
                        color="rgba(0, 229, 255, 0.15)", 
                        line=dict(color="#00e5ff", width=2)
                    ),
                    text=white_mistakes, 
                    textposition="outside",
                    textfont=dict(color="#e6edf3", size=12, weight="bold"),
                    hovertemplate="White %{x}: %{y}<extra></extra>"
                ))

                fig5.update_layout(
                    barmode="group",
                    height=380,
                    hoverlabel=dict(
                        bgcolor="rgba(10, 20, 45, 0.9)",
                        bordercolor="rgba(96, 165, 250, 0.5)",
                        font=dict(color="#e6edf3", size=12, family="system-ui, sans-serif")
                    ),
                    xaxis=dict(
                        title="Game phase",
                        tickfont=dict(color="#e6edf3", size=12, family="system-ui, sans-serif"),
                        gridcolor="rgba(0,0,0,0)",
                    ),
                    yaxis=dict(
                        title="Mistakes + Inaccuracies + Blunders",
                        gridcolor="rgba(125,143,168,0.08)",
                        color="#7d8fa8",
                        zeroline=False,
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                        font=dict(color="#c9d1d9", size=12),
                        bgcolor="rgba(10,20,45,0.75)",
                        bordercolor="rgba(96,165,250,0.35)",
                        borderwidth=1,
                    ),
                    margin=dict(l=50, r=20, t=60, b=50),
                    plot_bgcolor="rgba(5, 15, 35, 0.35)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9d1d9", family="system-ui, sans-serif"),
                )

                import plotly.io as pio
                
                _post_md = """
                setTimeout(function() {
                    var plot = document.querySelector('.js-plotly-plot');
                    if (!plot) return;
                    
                    // Apply glow to bar outlines
                    var bars = plot.querySelectorAll('.barlayer .point path');
                    bars.forEach(function(b) {
                        var stroke = b.getAttribute('stroke');
                        if (stroke === '#ff2a2a' || stroke.indexOf('rgb(255, 42, 42)') > -1) {
                            b.style.filter = 'drop-shadow(0 0 8px rgba(255, 42, 42, 0.6))';
                        } else if (stroke === '#00e5ff' || stroke.indexOf('rgb(0, 229, 255)') > -1) {
                            b.style.filter = 'drop-shadow(0 0 8px rgba(0, 229, 255, 0.6))';
                        }
                    });

                    // Add Fullscreen button
                    var bar = document.querySelector('.modebar');
                    if (!bar || bar.querySelector('.fs-btn')) return;
                    var btn = document.createElement('a');
                    btn.className = 'modebar-btn fs-btn';
                    btn.title = 'Fullscreen';
                    btn.innerHTML =
                        '<svg viewBox="0 0 24 24">' +
                        '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5z' +
                        'm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
                    btn.addEventListener('click', function(e) {
                        e.preventDefault(); e.stopPropagation();
                        var gd = document.querySelector('.js-plotly-plot');
                        if (!document.fullscreenElement) {
                            if (gd.requestFullscreen) gd.requestFullscreen();
                            else if (gd.webkitRequestFullscreen) gd.webkitRequestFullscreen();
                            else if (gd.msRequestFullscreen) gd.msRequestFullscreen();
                        } else {
                            if (document.exitFullscreen) document.exitFullscreen();
                        }
                    });
                    bar.appendChild(btn);
                }, 500);
                """

                _md_html = _modebar_style + pio.to_html(
                    fig5,
                    include_plotlyjs='cdn',
                    full_html=False,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'responsive': True,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'toggleSpikelines'],
                        'toImageButtonOptions': {'format': 'png', 'filename': 'mistake_density', 'scale': 2},
                    },
                    post_script=_post_md,
                )
                components.html(_md_html, height=400, scrolling=False)

        # ---- Sub-tab 4: Explanations ----
        with subtab_explain:
            # Header
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "align-items:flex-start;margin-top:16px;margin-bottom:24px;'>"
                "<div style='display:flex;align-items:center;gap:14px;'>"
                "<div style='width:44px;height:44px;border-radius:12px;"
                "background:linear-gradient(135deg,#2563eb,#7c3aed);"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(59,130,246,0.5);'>💡</div>"
                "<div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "Move <span style='background:linear-gradient(90deg,#3b82f6 0%,"
                "#22d3ee 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "Explanations</span></div>"
                "<div style='color:#7d8fa8;font-size:0.95rem;margin-top:2px;'>"
                "Understand why each move is good or not — and how many pawns it cost.</div>"
                "</div></div>"
                "</div>",
                unsafe_allow_html=True)

            best_alts = graph_data.get("best_alts", [])
            cats = graph_data.get("categories", [])
            cpls_all = graph_data.get("cpls", [])
            sans_all = graph_data.get("sans", [])
            fens_all = graph_data.get("fens", [])
            
            highlights = []
            for i in range(1, len(sans_all)):
                if cats[i] in ("Mistake", "Blunder", "Inaccuracy"):
                    highlights.append({
                        "ply": i,
                        "move_number": (i + 1) // 2,
                        "color": "White" if i % 2 == 1 else "Black",
                        "san": sans_all[i],
                        "category": cats[i],
                        "cpl": cpls_all[i],
                        "best_alt": best_alts[i] if i < len(best_alts) else "",
                        "fen_before": fens_all[i - 1] if i - 1 < len(fens_all) else "",
                    })

            if not highlights:
                st.success("No mistakes or blunders in this game! Excellent play.")
            else:
                # Filter toggle
                color_filter = st.radio(
                    "Filter by player",
                    ["All", "White", "Black"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="explain_filter"
                )

                if color_filter != "All":
                    highlights = [h for h in highlights if h["color"] == color_filter]

                # Build the HTML string for all cards
                cards_html = ""
                for h in highlights:
                    explanation = explain_move(h, h["fen_before"])
                    
                    # Move number badge styling
                    is_white = h["color"] == "White"
                    badge_bg = "linear-gradient(135deg, #2563eb, #3b82f6)" if is_white else "linear-gradient(135deg, #991b1b, #ef4444)"
                    badge_glow = "rgba(59,130,246,0.4)" if is_white else "rgba(239,68,68,0.4)"
                    
                    # Piece icon
                    san_start = h["san"][0] if h["san"] else ""
                    piece_icon = "♟"
                    if san_start == "N": piece_icon = "♞"
                    elif san_start == "B": piece_icon = "♝"
                    elif san_start == "R": piece_icon = "♜"
                    elif san_start == "Q": piece_icon = "♛"
                    elif san_start == "K": piece_icon = "♚"
                    elif is_white: piece_icon = "♙"
                    
                    # Classification Badge styling
                    cat = h["category"]
                    if cat == "Mistake":
                        cat_icon, cat_color = "❌", "#ef4444"
                    elif cat == "Blunder":
                        cat_icon, cat_color = "💥", "#ec4899"
                    else:
                        cat_icon, cat_color = "⚠️", "#f97316"
                        
                    # Card HTML
                    cards_html += (
                        f"<div style='background:rgba(10,20,45,0.7);"
                        f"border:1px solid rgba(96,165,250,0.2);"
                        f"border-radius:12px;padding:16px 20px;"
                        f"margin-bottom:12px;display:flex;"
                        f"align-items:center;gap:18px;"
                        f"transition: all 0.2s ease;'>"
                        
                        # Left: Move number badge
                        f"<div style='min-width:40px;height:40px;border-radius:8px;"
                        f"background:{badge_bg};display:flex;align-items:center;"
                        f"justify-content:center;font-weight:800;font-size:1.1rem;"
                        f"color:white;box-shadow:0 0 12px {badge_glow};'>"
                        f"{h['move_number']}</div>"
                        
                        # Middle: Move info
                        f"<div style='flex-grow:1;'>"
                        # Top row
                        f"<div style='display:flex;align-items:center;gap:12px;"
                        f"margin-bottom:6px;'>"
                        f"<span style='color:#e6edf3;font-weight:700;font-size:1.05rem;'>"
                        f"Move {h['move_number']} ({h['color']}): {h['san']}</span>"
                        f"<span style='color:#7d8fa8;font-size:1.2rem;'>{piece_icon}</span>"
                        f"<span style='display:inline-flex;align-items:center;gap:6px;"
                        f"background:rgba(0,0,0,0.2);border:1px solid {cat_color}44;"
                        f"color:{cat_color};padding:4px 10px;border-radius:20px;"
                        f"font-size:0.8rem;font-weight:700;'>"
                        f"{cat_icon} {cat} (lost {h['cpl']} cp)</span>"
                        f"</div>"
                        # Bottom row
                        f"<div style='color:#8b9bb4;font-size:0.9rem;'>"
                        f"→ {explanation}</div>"
                        f"</div>"
                        
                        # Right: Chevron
                        f"<div style='color:#4b5563;font-size:1.2rem;'>›</div>"
                        f"</div>"
                    )
                
                st.markdown(cards_html, unsafe_allow_html=True)
                
                # Bottom info box
                st.markdown(
                    "<div style='background:rgba(10,20,45,0.8);"
                    "border:1px solid rgba(96,165,250,0.3);border-radius:12px;"
                    "padding:20px;display:flex;align-items:flex-start;gap:16px;"
                    "margin-top:20px;'>"
                    "<div style='width:40px;height:40px;border-radius:50%;"
                    "background:rgba(59,130,246,0.15);display:flex;"
                    "align-items:center;justify-content:center;font-size:1.2rem;'>📈</div>"
                    "<div>"
                    "<div style='color:#e6edf3;font-weight:700;font-size:1.05rem;margin-bottom:4px;'>"
                    "These moves cost significant material.</div>"
                    "<div style='color:#7d8fa8;font-size:0.9rem;'>"
                    "Focus on developing your pieces and avoiding unnecessary risks.</div>"
                    "</div></div>",
                    unsafe_allow_html=True
                )

        # ---- Sub-tab 5: Browser ----
        with subtab_browse:
            # 1. Header
            st.markdown(
                "<div style='display:flex;align-items:center;gap:14px;"
                "margin-top:16px;margin-bottom:24px;'>"
                "<div style='width:44px;height:44px;border-radius:12px;"
                "background:linear-gradient(135deg,#2563eb,#7c3aed);"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:1.3rem;"
                "box-shadow:0 0 16px rgba(59,130,246,0.5);'>♞</div>"
                "<div>"
                "<div style='font-size:1.8rem;font-weight:800;"
                "letter-spacing:-0.5px;color:#e6edf3;'>"
                "Position <span style='background:linear-gradient(90deg,#3b82f6 0%,"
                "#22d3ee 100%);-webkit-background-clip:text;"
                "background-clip:text;-webkit-text-fill-color:transparent;'>"
                "Browser</span></div>"
                "<div style='color:#7d8fa8;font-size:0.95rem;margin-top:2px;'>"
                "Explore and analyze any position in the game</div>"
                "</div></div>",
                unsafe_allow_html=True)

            fens = graph_data.get("fens", [])
            sans = graph_data.get("sans", [])
            evals = graph_data.get("evals", [])

            if len(fens) <= 1:
                st.info("No moves to browse — the game appears to be empty or could not be parsed.")
            else:
                if "current_review_ply" not in st.session_state:
                    st.session_state.current_review_ply = 0
                    
                

                # 3. Scatter Points Toggle
                show_dots = st.toggle("Show Scatter Points", value=False, key="browser_scatter_toggle")
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

                col_board, col_info = st.columns([1, 1.2], gap="medium")

                

                with col_board:
                    with st.container(border=True):
                        # 5. Horizontal Move Timeline
                        st.markdown(
                            "<div style='color:#c9d1d9;font-weight:600;"
                            "font-size:0.9rem;margin-bottom:4px;'>Jump to Move</div>",
                            unsafe_allow_html=True)
                        
                        move_labels = ["Start"]
                        for i, s in enumerate(sans[1:]):
                            if i % 2 == 0:
                                move_labels.append(f"🔵 {s}")
                            else:
                                move_labels.append(f"🔴 {s}")
                                
                        selected_move_label = st.radio(
                            "Move Timeline",
                            move_labels,
                            index=st.session_state.current_review_ply,
                            horizontal=True,
                            label_visibility="collapsed",
                            key="move_timeline_radio"
                        )
                        
                        new_ply = move_labels.index(selected_move_label)
                        if new_ply != st.session_state.current_review_ply:
                            st.session_state.current_review_ply = new_ply
                            st.rerun()
                            
                        st.markdown("<hr style='margin:10px 0;border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

                        # 6. Slider
                        st.markdown(
                            "<div style='color:#c9d1d9;font-weight:600;"
                            "font-size:0.9rem;margin-bottom:10px;'>"
                            "Select ply (half-move)</div>",
                            unsafe_allow_html=True)
                            
                        selected_ply = st.slider(
                            "Select ply (half-move)",
                            min_value=0, max_value=len(fens) - 1,
                            value=st.session_state.current_review_ply,
                            label_visibility="collapsed")
                            
                        if selected_ply != st.session_state.current_review_ply:
                            st.session_state.current_review_ply = selected_ply

                        # 7. Board
                        if selected_ply == 0:
                            label = "Starting position"
                        else:
                            move_num = (selected_ply + 1) // 2
                            side = "White" if selected_ply % 2 == 1 else "Black"
                            label = f"Move {move_num} ({side}): {sans[selected_ply]}"
                            
                        st.markdown(
                            f"<div style='color:#e6edf3;font-weight:700;"
                            f"font-size:1.1rem;margin-top:10px;margin-bottom:10px;'>"
                            f"{label}</div>",
                            unsafe_allow_html=True)

                        try:
                            board_obj = chess.Board(fens[selected_ply])
                            svg_str = chess.svg.board(board_obj, size=400)
                            svg_b64 = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
                            st.markdown(
                                f"<div style='border-radius:10px;overflow:hidden;"
                                f"box-shadow:0 0 20px rgba(59,130,246,0.3);'>"
                                f"<img src='data:image/svg+xml;base64,{svg_b64}' width='100%' />"
                                f"</div>",
                                unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Could not render board: {e}")

                        # 8. Mini Evaluation Graph
                        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
                        st.markdown(
                            "<div style='color:#c9d1d9;font-weight:600;"
                            "font-size:0.9rem;margin-bottom:4px;'>Position Highlight</div>",
                            unsafe_allow_html=True)
                        
                        fig_mini = go.Figure()
                        y_evals = [max(-800, min(800, v)) for v in evals]
                        white_adv = [max(0, v) for v in y_evals]
                        black_adv = [min(0, v) for v in y_evals]
                        x_vals = list(range(len(y_evals)))
                        

                        mode_val = "lines+markers" if show_dots else "lines"
                        fig_mini.add_trace(go.Scatter(
                            x=x_vals, y=white_adv, mode=mode_val,
                            line=dict(color="#3b82f6", width=2),
                            marker=dict(size=4, color="#3b82f6", line=dict(width=1, color="#fff")),
                            fill="tozeroy", fillcolor="rgba(59,130,246,0.2)",
                            hoverinfo="skip"
                        ))
                        fig_mini.add_trace(go.Scatter(
                            x=x_vals, y=black_adv, mode=mode_val,
                            line=dict(color="#ef4444", width=2),
                            marker=dict(size=4, color="#ef4444", line=dict(width=1, color="#fff")),
                            fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
                            hoverinfo="skip"
                        ))
                        
                        current_y = y_evals[selected_ply]
                        fig_mini.add_trace(go.Scatter(
                            x=[selected_ply], y=[current_y], mode="markers",
                            marker=dict(size=15, color="#00e5ff", 
                                        line=dict(color="#ffffff", width=2),
                                        symbol="circle"),
                            hoverinfo="skip"
                        ))
                        
                        fig_mini.add_vline(x=selected_ply, line_dash="dash", 
                                           line_color="#00e5ff", line_width=2,
                                           opacity=0.5)

                        fig_mini.update_layout(
                            height=150,
                            margin=dict(l=0, r=0, t=10, b=0),
                            plot_bgcolor="rgba(5, 15, 35, 0.35)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                            showlegend=False
                        )
                        
                        import plotly.io as pio
                        st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False})
                with col_info:
                    # Card 1: FEN
                    with st.container(border=True):
                        st.markdown(
                            "<div style='display:flex;align-items:center;gap:8px;"
                            "color:#60a5fa;font-weight:700;font-size:0.85rem;"
                            "text-transform:uppercase;margin-bottom:8px;'>"
                            "<span style='font-size:1.1rem;'>&lt;/&gt;</span> FEN:</div>",
                            unsafe_allow_html=True)
                        st.code(fens[selected_ply], language="text")

                    # Card 2: Evaluation
                    with st.container(border=True):
                        current_eval = evals[selected_ply]
                        
                        if current_eval >= 200: eval_color = "#22c55e"
                        elif current_eval <= -200: eval_color = "#ef4444"
                        elif abs(current_eval) < 50: eval_color = "#8b949e"
                        else: eval_color = "#facc15"

                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;"
                            f"color:#c9d1d9;font-weight:700;font-size:0.9rem;"
                            f"text-transform:uppercase;margin-bottom:12px;'>"
                            f"<span style='font-size:1.1rem;color:#60a5fa;'>★</span> "
                            f"Evaluation at this position</div>",
                            unsafe_allow_html=True)
                            
                        st.markdown(
                            f"<div style='color:{eval_color};font-size:1.8rem;"
                            f"font-weight:800;'>📊 {current_eval:+d} cp</div>",
                            unsafe_allow_html=True)
                            
                        if "inline_analysis" in st.session_state:
                            inline_result = st.session_state["inline_analysis"]
                            if inline_result.get("ply_warning"):
                                st.warning(inline_result["ply_warning"])

                    # Card 3: Position Analysis
                    with st.container(border=True):
                        st.markdown(
                            "<div style='display:flex;align-items:center;gap:8px;"
                            "color:#c9d1d9;font-weight:700;font-size:0.9rem;"
                            "text-transform:uppercase;margin-bottom:12px;'>"
                            "<span style='font-size:1.1rem;color:#3b82f6;'>📈</span> "
                            "Position analysis</div>",
                            unsafe_allow_html=True)
                            
                        try:
                            mat_bal = material_balance_from_fen(fens[selected_ply])
                            w_prob = eval_to_win_prob(current_eval)
                            
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Material balance", f"{mat_bal:+d}")
                            mc2.metric("Stockfish eval", f"{current_eval:+d} cp")
                            mc3.metric("Ply", selected_ply)
                            
                            st.markdown(
                                f"<div style='margin-top:10px;'>"
                                f"<div style='color:#7d8fa8;font-size:0.75rem;margin-bottom:4px;'>Evaluation bar</div>"
                                f"<div style='display:flex;height:24px;border-radius:6px;"
                                f"overflow:hidden;background:#1a2332;'>"
                                f"<div style='width:{w_prob:.1%};background:#22c55e;"
                                f"display:flex;align-items:center;justify-content:center;"
                                f"color:#052e16;font-weight:700;font-size:0.75rem;'>{w_prob:.0%}</div>"
                                f"<div style='width:{1-w_prob:.1%};background:#1e293b;"
                                f"display:flex;align-items:center;justify-content:center;"
                                f"color:#7d8fa8;font-weight:700;font-size:0.75rem;'>{1-w_prob:.0%}</div>"
                                f"</div></div>",
                                unsafe_allow_html=True)
                        except Exception:
                            pass

                    # Card 4: Prediction
                    current_fen = fens[selected_ply]
                    cache_key = f"fen_analysis_{current_fen}"
                    
                    if cache_key not in st.session_state:
                        with st.spinner("Analyzing position..."):
                            st.session_state[cache_key] = predict_from_fen(current_fen, 2615, 2610)
                            
                    inline_result = st.session_state[cache_key]

                    if "error" not in inline_result:
                        with st.container(border=True):
                            st.markdown(
                                "<div style='display:flex;align-items:center;gap:8px;"
                                "color:#c9d1d9;font-weight:700;font-size:0.9rem;"
                                "text-transform:uppercase;margin-bottom:12px;'>"
                                "<span style='font-size:1.1rem;color:#ec4899;'>🎯</span> "
                                "Prediction</div>",
                                unsafe_allow_html=True)
                            
                            if inline_result["p_white"] >= inline_result["p_black"] and inline_result["p_white"] >= inline_result["p_draw"]:
                                gv, gl, gc = inline_result["p_white"], "White to win", "#22c55e"
                            elif inline_result["p_black"] >= inline_result["p_white"] and inline_result["p_black"] >= inline_result["p_draw"]:
                                gv, gl, gc = inline_result["p_black"], "Black to win", "#3b82f6"
                            else:
                                gv, gl, gc = inline_result["p_draw"], "Draw expected", "#a855f7"

                            g1, g2 = st.columns([1, 1.4])
                            with g1:
                                render_gauge(gv, gl, gc)
                            with g2:
                                st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
                                render_probability_bars(inline_result["p_white"], inline_result["p_draw"], inline_result["p_black"])

                    # Card 5: Material Balance Mini
                    with st.container(border=True):
                        st.markdown(
                            "<div style='display:flex;align-items:center;gap:8px;"
                            "color:#c9d1d9;font-weight:700;font-size:0.9rem;"
                            "text-transform:uppercase;margin-bottom:12px;'>"
                            "<span style='font-size:1.1rem;color:#00e5ff;'>⚖️</span> "
                            "Material Balance</div>",
                            unsafe_allow_html=True)
                        
                        material = [material_balance_from_fen(f) for f in fens]
                        x_mat = list(range(len(material)))
                        white_mat = [max(0, m) for m in material]
                        black_mat = [min(0, m) for m in material]

                        fig_mat_mini = go.Figure()
                        fig_mat_mini.add_trace(go.Scatter(
                            x=x_mat, y=white_mat, mode=mode_val,
                            line=dict(color="#00e5ff", width=2),
                            marker=dict(size=4, color="#00e5ff", line=dict(width=1, color="#fff")),
                            fill="tozeroy", fillcolor="rgba(0, 229, 255, 0.15)",
                            hoverinfo="skip"
                        ))
                        fig_mat_mini.add_trace(go.Scatter(
                            x=x_mat, y=black_mat, mode=mode_val,
                            line=dict(color="#ff2a2a", width=2),
                            marker=dict(size=4, color="#ff2a2a", line=dict(width=1, color="#fff")),
                            fill="tozeroy", fillcolor="rgba(255, 42, 42, 0.15)",
                            hoverinfo="skip"
                        ))
                        
                        current_mat = material[selected_ply]
                        fig_mat_mini.add_trace(go.Scatter(
                            x=[selected_ply], y=[current_mat], mode="markers",
                            marker=dict(size=12, color="#ffffff", line=dict(color="#00e5ff", width=2)),
                            hoverinfo="skip"
                        ))
                        fig_mat_mini.add_vline(x=selected_ply, line_dash="dash", line_color="#ffffff", opacity=0.5)

                        fig_mat_mini.update_layout(
                            height=150, margin=dict(l=0, r=0, t=10, b=0),
                            plot_bgcolor="rgba(5, 15, 35, 0.35)", paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False
                        )
                        
                        st.plotly_chart(fig_mat_mini, use_container_width=True, config={'displayModeBar': False})
                    # Card 6: King Safety Mini
                    with st.container(border=True):
                        st.markdown(
                            "<div style='display:flex;align-items:center;gap:8px;"
                            "color:#c9d1d9;font-weight:700;font-size:0.9rem;"
                            "text-transform:uppercase;margin-bottom:12px;'>"
                            "<span style='font-size:1.1rem;color:#f59e0b;'>👑</span> "
                            "King Safety</div>",
                            unsafe_allow_html=True)
                        
                        w_safety, b_safety = king_safety_timeline(fens)
                        x_ks = list(range(len(w_safety)))

                        fig_ks_mini = go.Figure()
                        fig_ks_mini.add_trace(go.Scatter(
                            x=x_ks, y=w_safety, mode=mode_val,
                            line=dict(color="#00e5ff", width=2),
                            marker=dict(size=4, color="#00e5ff", line=dict(width=1, color="#fff")),
                            fill="tozeroy", fillcolor="rgba(0, 229, 255, 0.15)",
                            hoverinfo="skip"
                        ))
                        fig_ks_mini.add_trace(go.Scatter(
                            x=x_ks, y=b_safety, mode=mode_val,
                            line=dict(color="#ff2a2a", width=2),
                            marker=dict(size=4, color="#ff2a2a", line=dict(width=1, color="#fff")),
                            fill="tozeroy", fillcolor="rgba(255, 42, 42, 0.15)",
                            hoverinfo="skip"
                        ))
                        
                        fig_ks_mini.add_vline(x=selected_ply, line_dash="dash", line_color="#ffffff", opacity=0.5)

                        fig_ks_mini.update_layout(
                            height=150, margin=dict(l=0, r=0, t=10, b=0),
                            plot_bgcolor="rgba(5, 15, 35, 0.35)", paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False
                        )
                        
                        st.plotly_chart(fig_ks_mini, use_container_width=True, config={'displayModeBar': False})
# ---------- Sidebar ----------
st.sidebar.markdown(
    """
    <div style="
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        padding:24px 0 8px 0;
        text-align:center;
    ">
        <div style="
            font-size:96px;
            line-height:1;
            background: linear-gradient(135deg, #22c55e 0%, #a855f7 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 18px rgba(34,197,94,0.55))
                    drop-shadow(0 0 28px rgba(168,85,247,0.35));
            font-family: 'Segoe UI Symbol', serif;
        ">♔</div>
        <div style="
            margin-top:14px;
            font-size:1.15rem;
            font-weight:800;
            letter-spacing:0.5px;
            background: linear-gradient(90deg, #22c55e, #a855f7);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        ">CHESS PREDICTOR</div>
        <div style="
            margin-top:6px;
            color:#7d8fa8;
            font-size:0.78rem;
            letter-spacing:0.3px;
        ">ML-powered analysis</div>
    </div>

    <div style="
        margin-top:20px;
        padding:14px;
        background:#0d1420;
        border:1px solid #1f2b3d;
        border-radius:12px;
        color:#c9d1d9;
        font-size:0.78rem;
        line-height:1.6;
    ">
        <div style="
            color:#22c55e;
            font-weight:700;
            font-size:0.72rem;
            text-transform:uppercase;
            letter-spacing:1px;
            margin-bottom:8px;
        ">About</div>
        Built with <b style="color:#e6edf3;">python-chess</b>,
        <b style="color:#e6edf3;">Stockfish 19</b>, and
        <b style="color:#e6edf3;">scikit-learn</b>.<br><br>
        Trained on <b style="color:#e6edf3;">5,000 master games</b>
        (2600+ ELO).<br><br>
        Test accuracy:
        <b style="color:#22c55e;">~89%</b>
    </div>
    """,
    unsafe_allow_html=True,
)