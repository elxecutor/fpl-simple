from __future__ import annotations

from typing import List, Dict, Tuple, Iterable
from collections import defaultdict
from itertools import combinations
from .models import Player, Fixture, MIN_MINUTES_THRESHOLD, MIN_APPEARANCES_THRESHOLD

def calculate_fixture_difficulty_score(player: Player, next_n_gw: int = 5) -> float:
    """
    Calculate a multiplier based on upcoming fixture difficulty.
    Lower difficulty -> Higher multiplier.
    
    Base multiplier is 1.0.
    Each easy fixture (difficulty <= 2) adds bonus.
    Each hard fixture (difficulty >= 4) adds penalty.
    """
    if not player.upcoming_fixtures:
        return 1.0
        
    # Get next N fixtures
    fixtures = player.upcoming_fixtures[:next_n_gw]
    if not fixtures:
        return 1.0
        
    total_difficulty = 0
    count = 0
    
    for f in fixtures:
        # Determine if home or away for this player
        if f.team_h == player.team_id:
            diff = f.team_h_difficulty
        else:
            diff = f.team_a_difficulty
        total_difficulty += diff
        count += 1
        
    avg_diff = total_difficulty / count if count > 0 else 3.0
    
    # Formula (30% impact per difficulty step): 
    # Avg diff 3 -> 1.0
    # Avg diff 2 -> 1.3 (30% boost)
    # Avg diff 4 -> 0.7 (30% penalty)
    # Avg diff 1 -> 1.6 (60% boost)
    # Avg diff 5 -> 0.4 (60% penalty)

    multiplier = 1.0 + (3.0 - avg_diff) * 0.3
    return max(0.2, multiplier)  # Cap at 0.2 floor to emphasize fixture impact

def calculate_fixture_count_multiplier(player: Player, min_fixtures: int = 5) -> float:
    """
    Penalize players with very few upcoming fixtures (avoid short-term signings).
    Returns a multiplier: 1.0 for 5+ fixtures, steep penalties for fewer.
    
    Ideal: 5+ fixtures ahead
    4 fixtures: 0.80 (20% penalty)
    3 fixtures: 0.50 (50% penalty) 
    2 fixtures: 0.25 (75% penalty)
    1 fixture: 0.10 (90% penalty)
    """
    count = len(player.upcoming_fixtures)
    if count >= min_fixtures:
        return 1.0
    if count == 4:
        return 0.80
    if count == 3:
        return 0.50
    if count == 2:
        return 0.25
    if count == 1:
        return 0.10
    return 0.01  # No fixtures = almost worthless

def calculate_form_vs_season_metric(player: Player) -> float:
    """
    Compare recent form (last 30 days avg) vs season PPG.
    Returns the difference. Positive means overperforming season avg.
    """
    return player.form - player.points_per_game


def calculate_form_multiplier(player: Player) -> float:
    """Convert form delta into a lighter multiplier to avoid chasing recent spikes.
    Recent form (last 5 games) can be misleading; season consistency is more reliable.
    """
    diff = calculate_form_vs_season_metric(player)
    if diff >= 0:
        multiplier = 1.0 + diff * 0.06  # Lighter boost for good form
        return min(1.4, multiplier)
    multiplier = 1.0 + diff * 0.03  # Lighter penalty for dips
    return max(0.85, multiplier)


def calculate_dgw_multiplier(player: Player) -> float:
    """Boost players with DGW, penalize BGW players."""
    dgw_count = player.dgw_fixture_count
    if dgw_count >= 2:
        return 1.5  # 50% boost for DGW
    elif dgw_count == 0:
        return 0.3  # 70% penalty for BGW (blank)
    return 1.0  # Normal GW


def calculate_price_momentum_bonus(player: Player, for_transfer_in: bool = True) -> float:
    """
    Price momentum bonus for transfers:
    - Buying risers: +5% (get them before price rise)
    - Buying fallers: -5% (price dropping for a reason)
    - Selling risers: -5% (hold value)
    - Selling fallers: +5% (sell before more drops)
    """
    momentum = player.price_momentum
    if for_transfer_in:
        return 1.0 + momentum * 0.05  # Buy risers, avoid fallers
    else:
        return 1.0 - momentum * 0.05  # Sell fallers, hold risers


def calculate_adjusted_score(player: Player) -> float:
    """Combined score for squad selection (not transfers)."""
    fix_mult = calculate_fixture_difficulty_score(player)
    form_mult = calculate_form_multiplier(player)
    count_mult = calculate_fixture_count_multiplier(player)
    dgw_mult = calculate_dgw_multiplier(player)
    chance = player.chance_of_playing_next_round if player.chance_of_playing_next_round is not None else 100
    chance_mult = chance / 100.0
    minutes_mult = min(1.0, player.average_minutes_per_appearance / 60.0) if player.average_minutes_per_appearance > 0 else 0.5
    return player.composite_score * fix_mult * form_mult * count_mult * dgw_mult * chance_mult * minutes_mult

def analyze_chip_timing(fixtures: List[Fixture], next_n_gw: int = 10) -> List[Dict]:
    """
    Analyze upcoming gameweeks for Chip usage.
    Returns a list of GW analysis dicts.
    """
    # Group fixtures by event
    gw_fixtures = defaultdict(list)
    for f in fixtures:
        if f.event and not f.finished:
            gw_fixtures[f.event].append(f)
            
    analysis = []
    sorted_events = sorted(gw_fixtures.keys())[:next_n_gw]
    
    for event in sorted_events:
        ev_fixtures = gw_fixtures[event]
        
        # Count teams playing
        teams_playing = set()
        for f in ev_fixtures:
            teams_playing.add(f.team_h)
            teams_playing.add(f.team_a)
            
        num_teams = len(teams_playing)
        num_fixtures = len(ev_fixtures)
        
        # Detect Blanks and Doubles
        # Standard GW has 10 fixtures, 20 teams.
        is_blank = num_teams < 20
        is_double = num_fixtures > 10 # Rough heuristic, better is to check if any team plays twice
        
        # Check for specific teams playing twice
        team_counts = defaultdict(int)
        for f in ev_fixtures:
            team_counts[f.team_h] += 1
            team_counts[f.team_a] += 1
            
        dgw_teams = [t for t, c in team_counts.items() if c > 1]
        bgw_teams = [t for t in range(1, 21) if t not in teams_playing] # Assuming 20 teams
        
        # Calculate average fixture difficulty
        total_diff = 0
        count = 0
        for f in ev_fixtures:
            total_diff += f.team_h_difficulty + f.team_a_difficulty
            count += 2
        avg_diff = total_diff / count if count > 0 else 3.0
        
        score = 0
        reason = []
        
        if dgw_teams:
            score += len(dgw_teams) * 2
            reason.append(f"DGW: {len(dgw_teams)} teams play twice")
        
        if len(bgw_teams) > 4:
            score += 5 # Good for Free Hit if many blanks
            reason.append(f"BGW: {len(bgw_teams)} teams blank")
            
        # Add fixture difficulty score: lower avg_diff -> higher score (easier GW)
        diff_score = max(0, (5 - avg_diff) * 2)  # Bonus for easy fixtures
        score += diff_score
        if diff_score > 0:
            reason.append(f"Easy fixtures (avg diff {avg_diff:.1f})")
        
        analysis.append({
            "event": event,
            "score": score,
            "dgw_teams_count": len(dgw_teams),
            "bgw_teams_count": len(bgw_teams),
            "avg_difficulty": avg_diff,
            "reason": ", ".join(reason) if reason else "Standard GW"
        })
        
    return sorted(analysis, key=lambda x: x["score"], reverse=True)

def optimize_budget(
    current_squad: List[Player], 
    all_players: List[Player], 
    bank: float, 
    max_transfers: int = 1
) -> List[Dict]:
    """
    Suggest transfers to maximize score within budget.
    Uses full combinatorial search for optimal results.
    Includes: DGW/BGW awareness, price momentum, fixture difficulty.
    """
    max_transfers = max(1, min(max_transfers, 5))

    # Filter candidates
    candidates = []
    for p in all_players:
        chance = p.chance_of_playing_next_round
        if (
            (chance is None or chance >= 75)
            and p.average_minutes_per_appearance >= MIN_MINUTES_THRESHOLD
            and p.appearances >= MIN_APPEARANCES_THRESHOLD
        ):
            candidates.append(p)
    
    def get_transfer_score_in(p: Player) -> float:
        """Score for player being transferred IN."""
        adj_score = calculate_adjusted_score(p)
        fix_mult = calculate_fixture_difficulty_score(p)
        dgw_mult = calculate_dgw_multiplier(p)
        price_mult = calculate_price_momentum_bonus(p, for_transfer_in=True)
        # Double fixture difficulty for transfer decisions
        fix_boost = adj_score * (1.0 + (fix_mult - 1.0) * 2.0)
        # xP bonus for high expected points
        xp_bonus = max(0, p.expected_points_next - 4) * 5
        return fix_boost * dgw_mult * price_mult + xp_bonus
    
    def get_transfer_score_out(p: Player) -> float:
        """Score for player being transferred OUT (includes sell momentum)."""
        adj_score = calculate_adjusted_score(p)
        fix_mult = calculate_fixture_difficulty_score(p)
        dgw_mult = calculate_dgw_multiplier(p)
        price_mult = calculate_price_momentum_bonus(p, for_transfer_in=False)
        fix_boost = adj_score * (1.0 + (fix_mult - 1.0) * 2.0)
        return fix_boost * dgw_mult * price_mult
    
    player_scores = {p.id: get_transfer_score_in(p) for p in candidates}
    current_scores = {p.id: get_transfer_score_out(p) for p in current_squad}

    current_squad_ids = {p.id for p in current_squad}
    team_counts_base = defaultdict(int)
    for p in current_squad:
        team_counts_base[p.team_id] += 1
    
    # Build positional candidate pools - top players per position, sorted by score
    positional_candidates: Dict[int, List[Player]] = defaultdict(list)
    for p in candidates:
        if p.id not in current_squad_ids:
            positional_candidates[p.element_type].append(p)
    
    TOP_PER_POSITION = 30  # Increased for better coverage
    for pos, plist in positional_candidates.items():
        plist.sort(key=lambda p: player_scores.get(p.id, 0), reverse=True)
        positional_candidates[pos] = plist[:TOP_PER_POSITION]
    
    recommendations = []
    
    def find_optimal_ins(out_players: List[Player]) -> List[Dict]:
        """
        For a given set of OUT players, find the optimal IN combination.
        Uses recursive search with pruning for efficiency.
        """
        n = len(out_players)
        total_budget = bank + sum(p.selling_price / 10 for p in out_players)
        out_score = sum(current_scores[p.id] for p in out_players)
        
        # Build team counts after removing OUT players
        team_counts = team_counts_base.copy()
        for p in out_players:
            team_counts[p.team_id] -= 1
        
        # Group OUT players by position to know what we need to fill
        positions_needed = [p.element_type for p in out_players]
        
        # Get candidate pools for each position needed
        candidate_pools = []
        for pos in positions_needed:
            pool = [p for p in positional_candidates.get(pos, []) if p.id not in current_squad_ids]
            if not pool:
                return []  # No candidates for this position
            candidate_pools.append(pool)
        
        best_result = None
        best_delta = 0
        
        def search(idx: int, chosen: List[Player], remaining_budget: float, temp_counts: Dict, used_ids: set):
            nonlocal best_result, best_delta
            
            if idx == n:
                # Complete combination found
                in_score = sum(player_scores.get(p.id, 0) for p in chosen)
                delta = in_score - out_score
                if delta > best_delta:
                    best_delta = delta
                    best_result = {
                        "type": f"{n}_transfer",
                        "out": list(out_players),
                        "in": list(chosen),
                        "delta": delta,
                        "out_score": out_score,
                        "in_score": in_score
                    }
                return
            
            # Pruning: calculate maximum possible remaining score
            # (sum of best available player in each remaining position)
            max_remaining = 0
            for future_idx in range(idx, n):
                future_pool = [p for p in candidate_pools[future_idx] 
                               if p.id not in used_ids and (p.now_cost / 10) <= remaining_budget]
                if future_pool:
                    max_remaining += max(player_scores.get(p.id, 0) for p in future_pool)
            
            current_in_score = sum(player_scores.get(p.id, 0) for p in chosen)
            if current_in_score + max_remaining - out_score <= best_delta:
                return  # Prune: can't beat best
            
            # Try candidates for this position
            for cand in candidate_pools[idx]:
                cost = cand.now_cost / 10
                if cost > remaining_budget:
                    continue
                if cand.id in used_ids:
                    continue
                if temp_counts.get(cand.team_id, 0) >= 3:
                    continue
                
                # Choose this candidate
                new_counts = temp_counts.copy()
                new_counts[cand.team_id] += 1
                new_used = used_ids | {cand.id}
                
                search(idx + 1, chosen + [cand], remaining_budget - cost, new_counts, new_used)
        
        search(0, [], total_budget, team_counts, current_squad_ids.copy())
        
        return [best_result] if best_result else []
    
    # Try ALL OUT combinations for each transfer count
    for n_transfers in range(1, max_transfers + 1):
        for out_combo in combinations(current_squad, n_transfers):
            results = find_optimal_ins(list(out_combo))
            recommendations.extend(results)
    
    # Sort by delta and return
    recommendations.sort(key=lambda x: x['delta'], reverse=True)
    return recommendations


def select_best_xi(squad: List[Player]) -> Dict:
    """
    Select the best starting XI based on AdjScore.
    Returns a dict with 'xi', 'subs', 'formation', 'score'.
    """
    # Calculate adjusted scores with tripled fixture difficulty weighting
    player_scores = {}
    for p in squad:
        adj_score = calculate_adjusted_score(p)
        fix_mult = calculate_fixture_difficulty_score(p)
        # Triple fixture difficulty impact, consistent with selection and optimization
        player_scores[p.id] = adj_score * (1.0 + (fix_mult - 1.0) * 3.0)

    # Separate by position
    gkps = [p for p in squad if p.element_type == 1]
    defs = [p for p in squad if p.element_type == 2]
    mids = [p for p in squad if p.element_type == 3]
    fwds = [p for p in squad if p.element_type == 4]

    # Sort by adjusted score descending
    gkps.sort(key=lambda p: player_scores[p.id], reverse=True)
    defs.sort(key=lambda p: player_scores[p.id], reverse=True)
    mids.sort(key=lambda p: player_scores[p.id], reverse=True)
    fwds.sort(key=lambda p: player_scores[p.id], reverse=True)

    # Valid formations (DEF-MID-FWD)
    formations = [
        (3, 5, 2), (3, 4, 3),
        (4, 5, 1), (4, 4, 2), (4, 3, 3),
        (5, 4, 1), (5, 3, 2), (5, 2, 3)
    ]

    best_xi = []
    best_score = -1.0
    best_formation = ""

    # Must have at least 1 GK
    if not gkps:
        return {}

    for n_def, n_mid, n_fwd in formations:
        # Check if we have enough players
        if len(defs) < n_def or len(mids) < n_mid or len(fwds) < n_fwd:
            continue

        current_xi = [gkps[0]] + defs[:n_def] + mids[:n_mid] + fwds[:n_fwd]
        # Score XI using base adjusted scores for display purposes
        current_score = sum(player_scores[p.id] for p in current_xi)

        if current_score > best_score:
            best_score = current_score
            best_xi = current_xi
            best_formation = f"{n_def}-{n_mid}-{n_fwd}"

    # Identify subs
    xi_ids = {p.id for p in best_xi}
    subs = [p for p in squad if p.id not in xi_ids]
    # Sort subs: GKP first, then by base adjusted score
    subs.sort(key=lambda p: (p.element_type != 1, -player_scores[p.id]))

    # Identify Captain and Vice Captain (based on base adjusted score)
    sorted_xi = sorted(best_xi, key=lambda p: player_scores[p.id], reverse=True)
    captain = sorted_xi[0] if sorted_xi else None
    vice_captain = sorted_xi[1] if len(sorted_xi) > 1 else None

    return {
        "xi": best_xi,
        "subs": subs,
        "formation": best_formation,
        "total_score": best_score,
        "player_scores": player_scores,
        "captain": captain,
        "vice_captain": vice_captain
    }


def select_budget_dream_xi(all_players: List[Player], budget: float = 100.0) -> Dict:
    """
    Select the best dream XI with a budget constraint (default 100m).
    Uses greedy selection prioritizing score while respecting budget.
    """
    candidates = [
        p for p in all_players
        if (p.chance_of_playing_next_round is None or p.chance_of_playing_next_round >= 75)
        and p.average_minutes_per_appearance >= MIN_MINUTES_THRESHOLD
        and p.appearances >= MIN_APPEARANCES_THRESHOLD
    ]
    
    player_scores = {}
    player_values = {}
    for p in candidates:
        adj_score = calculate_adjusted_score(p)
        fix_mult = calculate_fixture_difficulty_score(p)
        fix_boost = adj_score * (1.0 + (fix_mult - 1.0) * 3.0)
        xp_bonus = max(0, p.expected_points_next - 4) * 5
        player_scores[p.id] = fix_boost + xp_bonus
        player_values[p.id] = player_scores[p.id] / (p.now_cost / 10) if p.now_cost > 0 else 0
    
    sorted_candidates = sorted(candidates, key=lambda p: player_scores[p.id], reverse=True)
    
    selected = []
    pos_limits = {1: 2, 2: 5, 3: 5, 4: 3}
    pos_min = {1: 1, 2: 3, 3: 2, 4: 1}  # Minimum for valid XI
    team_counts = defaultdict(int)
    pos_counts = defaultdict(int)
    total_cost = 0.0
    
    # First pass: fill minimums with best value players
    for pos in [1, 2, 3, 4]:
        pos_cands = sorted([p for p in candidates if p.element_type == pos], 
                          key=lambda p: player_values.get(p.id, 0), 
                          reverse=True)
        for p in pos_cands:
            cost = p.now_cost / 10
            if (pos_counts[pos] < pos_min[pos] 
                and team_counts[p.team_id] < 3 
                and total_cost + cost <= budget):
                selected.append(p)
                pos_counts[pos] += 1
                team_counts[p.team_id] += 1
                total_cost += cost
                if pos_counts[pos] >= pos_min[pos]:
                    break
    
    # Second pass: fill remaining 11 spots with best available
    for p in sorted_candidates:
        if len(selected) >= 11:
            break
        if p in selected:
            continue
        cost = p.now_cost / 10
        if (pos_counts[p.element_type] < pos_limits[p.element_type]
            and team_counts[p.team_id] < 3
            and total_cost + cost <= budget):
            selected.append(p)
            pos_counts[p.element_type] += 1
            team_counts[p.team_id] += 1
            total_cost += cost
    
    return {
        "xi": selected,
        "formation": "Budget Dream XI",
        "total_score": sum(player_scores.get(p.id, 0) for p in selected),
        "total_cost": total_cost,
        "player_scores": player_scores,
    }


def find_differentials(all_players: List[Player], max_ownership: float = 10.0, top_n: int = 15) -> List[Dict]:
    """
    Find high-potential differential picks (low ownership, high score).
    """
    candidates = [
        p for p in all_players
        if (p.chance_of_playing_next_round is None or p.chance_of_playing_next_round >= 75)
        and p.average_minutes_per_appearance >= MIN_MINUTES_THRESHOLD
        and p.appearances >= MIN_APPEARANCES_THRESHOLD
        and p.selected_by_percent <= max_ownership
    ]
    
    results = []
    for p in candidates:
        adj_score = calculate_adjusted_score(p)
        fix_mult = calculate_fixture_difficulty_score(p)
        fix_boost = adj_score * (1.0 + (fix_mult - 1.0) * 3.0)
        xp_bonus = max(0, p.expected_points_next - 4) * 5
        score = fix_boost + xp_bonus
        # Differential bonus: lower ownership = higher bonus
        diff_bonus = (max_ownership - p.selected_by_percent) * 2
        results.append({
            "player": p,
            "score": score,
            "diff_score": score + diff_bonus,
            "ownership": p.selected_by_percent,
        })
    
    results.sort(key=lambda x: x["diff_score"], reverse=True)
    return results[:top_n]


def compare_players(player1: Player, player2: Player) -> Dict:
    """
    Compare two players head-to-head.
    """
    def get_metrics(p):
        adj_score = calculate_adjusted_score(p)
        fix_mult = calculate_fixture_difficulty_score(p)
        fix_boost = adj_score * (1.0 + (fix_mult - 1.0) * 3.0)
        return {
            "adj_score": fix_boost,
            "ict": p.ict_index,
            "ppg": p.points_per_game,
            "xp": p.expected_points_next,
            "form": p.form,
            "fix_mult": fix_mult,
            "cost": p.now_cost / 10,
            "ownership": p.selected_by_percent,
            "minutes": p.average_minutes_per_appearance,
            "value": fix_boost / (p.now_cost / 10) if p.now_cost > 0 else 0,
        }
    
    m1 = get_metrics(player1)
    m2 = get_metrics(player2)
    
    return {
        "player1": {"player": player1, "metrics": m1},
        "player2": {"player": player2, "metrics": m2},
        "winner": player1 if m1["adj_score"] > m2["adj_score"] else player2,
        "score_diff": abs(m1["adj_score"] - m2["adj_score"]),
    }


def calculate_squad_distance_from_dream(current_squad: List[Player], dream_xi: List[Player]) -> Dict:
    """
    Calculate how far current squad is from dream XI.
    """
    current_ids = {p.id for p in current_squad}
    dream_ids = {p.id for p in dream_xi}
    
    in_both = current_ids & dream_ids
    missing = [p for p in dream_xi if p.id not in current_ids]
    to_remove = [p for p in current_squad if p.id not in dream_ids]
    
    return {
        "overlap": len(in_both),
        "missing": missing,
        "to_remove": to_remove,
        "transfers_needed": len(missing),
    }
