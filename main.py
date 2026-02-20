from __future__ import annotations

import argparse
import unicodedata
import json
import os
from collections import Counter
from typing import Iterable, Sequence, List

from fpl_simple import FPLClient, select_best_squad
from fpl_simple.analysis import (
    calculate_fixture_difficulty_score, 
    calculate_form_multiplier,
    calculate_adjusted_score,
    analyze_chip_timing,
    optimize_budget,
    select_best_xi,
    select_budget_dream_xi,
    find_differentials,
    compare_players,
    calculate_squad_distance_from_dream,
    calculate_fixture_count_multiplier
)

def normalize_name(name: str) -> str:
    """Normalize string to handle accents and case."""
    n = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    return n.lower().strip()

def get_display_score(p):
    base = calculate_adjusted_score(p)
    fix = calculate_fixture_difficulty_score(p)
    return base * (1.0 + (fix - 1.0) * 3.0)

def format_row(player) -> str:
    fix_score = calculate_fixture_difficulty_score(player)
    count_mult = calculate_fixture_count_multiplier(player)
    form_mult = calculate_form_multiplier(player)

    # Adjusted score: ICT * PPG * Pos * Fix * Form * Count * Chance * Minutes
    adj_score = get_display_score(player)
    price_trend = player.price_trend
    
    return (
        f"{player.web_name:<18} {player.team_name:<12} {player.cost_str:>5}{price_trend:<1} "
        f"ICT {player.ict_index:>5.1f} | PPG {player.points_per_game:>4.1f} | xP {player.expected_points_next:>4.1f} | "
        f"Fix {fix_score:>4.2f} | Own {player.selected_by_percent:>5.1f}% | "
        f"AdjScore {adj_score:>7.1f}"
    )


def print_position_block(position: str, players: Sequence) -> None:
    print(f"\n{position} ({len(players)} selected)")
    print("-" * 90)
    for player in players:
        print(format_row(player))


def main() -> None:
    parser = argparse.ArgumentParser(description="FPL Squad Selector & Analyzer")
    parser.add_argument("--chips", action="store_true", help="Analyze best weeks for Chips")
    parser.add_argument("--optimize", action="store_true", help="Run budget optimizer (requires squad input)")
    parser.add_argument("--xi", action="store_true", help="Select best starting XI from squad")
    parser.add_argument("--dream", action="store_true", help="Select best dream XI within 100m budget")
    parser.add_argument("--diff", action="store_true", help="Find differential picks (low ownership, high potential)")
    parser.add_argument("--compare", type=str, help="Compare two players: --compare 'player1,player2'")
    parser.add_argument("--squad", type=str, help="Comma-separated list of player web_names for optimization")
    parser.add_argument("--file", type=str, default="squad.json", help="JSON file to load squad/bank from (default: squad.json)")
    parser.add_argument("--bank", type=float, default=None, help="Money in the bank (overrides file)")
    args = parser.parse_args()

    client = FPLClient()
    print("Fetching data...")
    players = client.fetch_players()
    
    # Load chips from file
    available_chips = []
    if args.file and os.path.exists(args.file):
        try:
            with open(args.file, 'r') as f:
                data = json.load(f)
                available_chips = data.get("chips", [])
        except Exception as e:
            print(f"Error loading chips from {args.file}: {e}")
    
    if args.chips:
        print("\n--- Chip Timing Analysis (Next 10 GWs) ---")
        fixtures = client.fetch_fixtures()
        analysis = analyze_chip_timing(fixtures)
        if not analysis:
            print("No upcoming fixtures found.")
            return
        
        # Recommend chips based on GW characteristics
        if available_chips:
            print(f"\nAvailable Chips: {', '.join(available_chips)}")
            print("\nRecommendations:")
            
            used_gws = set()
            
            # Free Hit: Best for BGW (many teams blanking)
            if "free_hit" in available_chips:
                bgw_gws = [gw for gw in analysis if gw['bgw_teams_count'] > 4]
                if bgw_gws:
                    best_bgw = max(bgw_gws, key=lambda x: x['bgw_teams_count'])
                    print(f"  Free Hit: GW {best_bgw['event']} - {best_bgw['bgw_teams_count']} teams blank")
                    used_gws.add(best_bgw['event'])
                else:
                    print("  Free Hit: No significant BGW found - save for later")
            
            # Bench Boost: Best for DGW (many teams playing twice)
            if "bench_boost" in available_chips:
                dgw_gws = [gw for gw in analysis if gw['dgw_teams_count'] > 0 and gw['event'] not in used_gws]
                if dgw_gws:
                    best_dgw = max(dgw_gws, key=lambda x: x['dgw_teams_count'])
                    print(f"  Bench Boost: GW {best_dgw['event']} - {best_dgw['dgw_teams_count']} teams have doubles")
                    used_gws.add(best_dgw['event'])
                else:
                    # Fall back to easiest fixtures GW
                    easy_gws = [gw for gw in analysis if gw['event'] not in used_gws]
                    if easy_gws:
                        best = min(easy_gws, key=lambda x: x['avg_difficulty'])
                        print(f"  Bench Boost: GW {best['event']} - Easy fixtures (avg diff {best['avg_difficulty']:.1f})")
                        used_gws.add(best['event'])
            
            # Triple Captain: Best for DGW or easy fixture week
            if "triple_captain" in available_chips:
                dgw_gws = [gw for gw in analysis if gw['dgw_teams_count'] > 0 and gw['event'] not in used_gws]
                if dgw_gws:
                    best_dgw = max(dgw_gws, key=lambda x: x['dgw_teams_count'])
                    print(f"  Triple Captain: GW {best_dgw['event']} - {best_dgw['dgw_teams_count']} teams have doubles")
                    used_gws.add(best_dgw['event'])
                else:
                    easy_gws = [gw for gw in analysis if gw['event'] not in used_gws]
                    if easy_gws:
                        best = min(easy_gws, key=lambda x: x['avg_difficulty'])
                        print(f"  Triple Captain: GW {best['event']} - Easy fixtures (avg diff {best['avg_difficulty']:.1f})")
                        used_gws.add(best['event'])
            
            # Wildcard: Strategic, recommend based on fixture swing
            if "wildcard" in available_chips:
                print("  Wildcard: Use before a fixture swing or to rebuild squad")
        else:
            print("No chips loaded from file.")
        
        print("\nGameweek Analysis:")
        for gw in sorted(analysis, key=lambda x: x['event']):
            flags = []
            if gw['dgw_teams_count'] > 0:
                flags.append(f"DGW({gw['dgw_teams_count']})")
            if gw['bgw_teams_count'] > 4:
                flags.append(f"BGW({gw['bgw_teams_count']})")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  GW {gw['event']}: Diff {gw['avg_difficulty']:.1f}{flag_str} - {gw['reason']}")
        return

    if args.dream:
        print("\n--- Dream XI (100m budget) ---")
        result = select_budget_dream_xi(players, budget=100.0)
        if not result or not result['xi']:
            print("Could not select a valid Budget Dream XI.")
        else:
            print(f"Formation: {result['formation']} | Total AdjScore: {result['total_score']:.1f} | Total Cost: {result['total_cost']:.1f}m")
            
            xi_gkp = [p for p in result['xi'] if p.element_type == 1]
            xi_def = [p for p in result['xi'] if p.element_type == 2]
            xi_mid = [p for p in result['xi'] if p.element_type == 3]
            xi_fwd = [p for p in result['xi'] if p.element_type == 4]
            
            for label, p_list in [("GKP", xi_gkp), ("DEF", xi_def), ("MID", xi_mid), ("FWD", xi_fwd)]:
                print(f"\n{label}:")
                for p in p_list:
                    score = result['player_scores'].get(p.id, 0)
                    print(f"  {p.web_name:<15} {p.team_name:<10} {p.cost_str:>5} Own {p.selected_by_percent:>5.1f}% AdjScore {score:.1f}")
            
            print("-" * 40)
        return

    if args.diff:
        print("\n--- Differential Picks (Ownership < 10%) ---")
        diffs = find_differentials(players, max_ownership=10.0, top_n=15)
        if not diffs:
            print("No differentials found.")
        else:
            print(f"{'Name':<18} {'Team':<12} {'Pos':<4} {'Cost':>5} {'Own':>6} {'xP':>4} {'AdjScore':>8}")
            print("-" * 70)
            for d in diffs:
                p = d['player']
                print(f"{p.web_name:<18} {p.team_name:<12} {p.position_name:<4} {p.cost_str:>5} {p.selected_by_percent:>5.1f}% {p.expected_points_next:>4.1f} {d['score']:>8.1f}")
        return

    if args.compare:
        names = [normalize_name(n.strip()) for n in args.compare.split(",")]
        if len(names) != 2:
            print("Error: --compare requires exactly two player names separated by comma")
            return
        
        p1 = p2 = None
        for p in players:
            if normalize_name(p.web_name) == names[0] or normalize_name(p.second_name) == names[0]:
                p1 = p
            if normalize_name(p.web_name) == names[1] or normalize_name(p.second_name) == names[1]:
                p2 = p
        
        if not p1:
            print(f"Could not find player: {names[0]}")
            return
        if not p2:
            print(f"Could not find player: {names[1]}")
            return
        
        result = compare_players(p1, p2)
        m1 = result['player1']['metrics']
        m2 = result['player2']['metrics']
        
        print(f"\n--- Player Comparison ---")
        print(f"{'Metric':<15} {p1.web_name:<15} {p2.web_name:<15} {'Winner':<10}")
        print("-" * 60)
        metrics = [
            ("AdjScore", m1['adj_score'], m2['adj_score']),
            ("ICT", m1['ict'], m2['ict']),
            ("PPG", m1['ppg'], m2['ppg']),
            ("xP", m1['xp'], m2['xp']),
            ("Form", m1['form'], m2['form']),
            ("Fixtures", m1['fix_mult'], m2['fix_mult']),
            ("Cost", m1['cost'], m2['cost']),
            ("Ownership", m1['ownership'], m2['ownership']),
            ("Value", m1['value'], m2['value']),
        ]
        for name, v1, v2 in metrics:
            if name == "Cost":
                winner = p1.web_name if v1 < v2 else p2.web_name if v2 < v1 else "Tie"
            elif name == "Ownership":
                winner = "-"  # Neither better/worse
            else:
                winner = p1.web_name if v1 > v2 else p2.web_name if v2 > v1 else "Tie"
            print(f"{name:<15} {v1:>14.1f} {v2:>14.1f} {winner:<10}")
        
        print(f"\n>>> Overall: {result['winner'].web_name} wins by {result['score_diff']:.1f} AdjScore")
        return

    if args.optimize or args.xi:
        print("\n--- Squad Analysis ---")
        
        squad_data = [] # List of (name, expected_type_id)
        bank = 0.0
        free_transfers = 1
        
        # Load from file if present
        if args.file and os.path.exists(args.file):
            try:
                with open(args.file, 'r') as f:
                    data = json.load(f)
                    if "squad" in data:
                        squad_data = [(normalize_name(n), None) for n in data["squad"]]
                    else:
                        # Try loading by position
                        type_map = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
                        for pos, type_id in type_map.items():
                            if pos in data:
                                squad_data.extend([(normalize_name(n), type_id) for n in data[pos]])
                                
                    if "bank" in data:
                        bank = float(data["bank"])
                    if "free_transfers" in data:
                        free_transfers = int(data["free_transfers"])
            except Exception as e:
                print(f"Error loading {args.file}: {e}")

        # Override with CLI args if provided
        if args.squad:
            squad_data = [(normalize_name(n), None) for n in args.squad.split(",")]
        if args.bank is not None:
            bank = args.bank

        if squad_data:
            print(f"Loaded squad of {len(squad_data)} players, {bank}m bank, {free_transfers} FT from {args.file}")

        if not squad_data:
            print("Error: No squad provided via --squad or --file.")
            return
        
        # Map names to player objects
        potential_matches = []
        for name, expected_type in squad_data:
            matches = []
            # Priority 1: Exact web_name match
            for p in players:
                if normalize_name(p.web_name) == name:
                    matches.append(p)
            
            # Priority 2: Exact second_name match
            if not matches:
                for p in players:
                    if normalize_name(p.second_name) == name:
                        matches.append(p)
            
            # Priority 3: Exact first_name match (risky)
            if not matches:
                for p in players:
                    if normalize_name(p.first_name) == name:
                        matches.append(p)
                        
            # Priority 4: Suffix match for "J.Timber"
            if not matches and "." in name:
                suffix = name.split(".")[-1]
                for p in players:
                    if normalize_name(p.second_name) == suffix:
                        matches.append(p)
            
            # Filter by expected type if provided
            if expected_type:
                matches = [p for p in matches if p.element_type == expected_type]

            potential_matches.append((name, matches))

        current_squad = []
        # Pre-calculate counts from unique matches
        temp_squad = [m[0] for n, m in potential_matches if len(m) == 1]
        counts = Counter(p.element_type for p in temp_squad)
        needed = {1: 2, 2: 5, 3: 5, 4: 3}

        for name, matches in potential_matches:
            if len(matches) == 1:
                current_squad.append(matches[0])
                if name not in [normalize_name(p.web_name) for p in temp_squad]: # Avoid double counting if already in temp_squad
                     counts[matches[0].element_type] += 1
            elif len(matches) > 1:
                best_match = matches[0]
                # Prefer the one that fills a needed slot
                for m in matches:
                    if counts[m.element_type] < needed.get(m.element_type, 0):
                        best_match = m
                        break
                current_squad.append(best_match)
                counts[best_match.element_type] += 1
            else:
                print(f"Warning: Could not find player '{name}'")
                
        if not current_squad:
            print("No valid players found in squad.")
            return
            
        # Validate squad structure (2 GKP, 5 DEF, 5 MID, 3 FWD)
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for p in current_squad:
            counts[p.element_type] += 1
            
        if counts[1] != 2 or counts[2] != 5 or counts[3] != 5 or counts[4] != 3:
            print("Warning: Squad does not follow standard structure (2 GKP, 5 DEF, 5 MID, 3 FWD).")
            print(f"Current: {counts[1]} GKP, {counts[2]} DEF, {counts[3]} MID, {counts[4]} FWD")
            
        if args.xi:
            print("\n--- Best Starting XI ---")
            result = select_best_xi(current_squad)
            if not result:
                print("Could not select a valid XI (check squad structure).")
            else:
                squad_value = sum(p.now_cost / 10 for p in current_squad)
                xi_value = sum(p.now_cost / 10 for p in result['xi'])
                print(f"Formation: {result['formation']} | Total AdjScore: {result['total_score']:.1f}")
                print(f"Squad Value: {squad_value:.1f}m | XI Value: {xi_value:.1f}m | Bank: {bank:.1f}m | Total: {squad_value + bank:.1f}m")
                
                # Show distance from dream XI
                dream = select_budget_dream_xi(players, budget=100.0)
                if dream and dream['xi']:
                    distance = calculate_squad_distance_from_dream(current_squad, dream['xi'])
                    print(f"Dream XI Overlap: {distance['overlap']}/15 players ({distance['transfers_needed']} transfers needed)")
                
                # Group XI by position for display
                xi_gkp = [p for p in result['xi'] if p.element_type == 1]
                xi_def = [p for p in result['xi'] if p.element_type == 2]
                xi_mid = [p for p in result['xi'] if p.element_type == 3]
                xi_fwd = [p for p in result['xi'] if p.element_type == 4]
                
                for label, p_list in [("GKP", xi_gkp), ("DEF", xi_def), ("MID", xi_mid), ("FWD", xi_fwd)]:
                    print(f"\n{label}:")
                    for p in p_list:
                        score = result['player_scores'][p.id]
                        role = ""
                        if result.get('captain') and p.id == result['captain'].id:
                            role = " (C)"
                        elif result.get('vice_captain') and p.id == result['vice_captain'].id:
                            role = " (VC)"
                        print(f"  {p.web_name:<15} {p.team_name:<10} {p.cost_str:>5} Own {p.selected_by_percent:>5.1f}% AdjScore {score:.1f}{role}")
                
                print("-" * 40)
                print("\nBENCH:")
                for p in result['subs']:
                    score = result['player_scores'][p.id]
                    print(f"  {p.web_name:<15} {p.team_name:<10} {p.cost_str:>5} Own {p.selected_by_percent:>5.1f}% AdjScore {score:.1f}")

        if args.optimize:
            print("\n--- Budget Optimizer ---")
            print(f"Optimizing for squad of {len(current_squad)} players with {bank}m bank...")
            max_transfer_eval = max(1, min(free_transfers, 5))
            recommendations = optimize_budget(
                current_squad,
                players,
                bank,
                max_transfers=max_transfer_eval
            )
            
            if not recommendations:
                print("No better transfers found within budget.")
                print("\n💡 Recommendation: ROLL your free transfer to next week.")
                return
            
            # Group by transfer count
            transfer_names = {1: "Single", 2: "Double", 3: "Triple", 4: "Quadruple", 5: "Quintuple"}
            
            # Filter to transfers matching FT count
            matching_transfers = [r for r in recommendations if len(r['out']) == free_transfers]
            
            # Calculate roll FT suggestion
            # If best delta is low, suggest rolling FT
            best_delta = recommendations[0]['delta'] if recommendations else 0
            best_single_delta = max((r['delta'] for r in recommendations if len(r['out']) == 1), default=0)
            
            # Threshold: if delta per transfer < 50, suggest rolling
            ROLL_THRESHOLD = 50
            delta_per_ft = best_delta / len(recommendations[0]['out']) if recommendations else 0
            should_roll = delta_per_ft < ROLL_THRESHOLD and free_transfers < 2
            
            if matching_transfers:
                print(f"\n=== Top 3 {transfer_names.get(free_transfers, str(free_transfers)+'-player')} Transfers (using all {free_transfers} FT) ===")
                for i, rec in enumerate(matching_transfers[:3]):
                    ft_cost = len(rec['out'])
                    print(f"{i+1}. {transfer_names.get(ft_cost, str(ft_cost)+'-player')} Transfer (Delta: {rec['delta']:.1f}):")
                    print("   OUT:")
                    for p_out in rec['out']:
                        print(f"     - {p_out.web_name} (Selling at {p_out.selling_cost_str}) - AdjScore {get_display_score(p_out):.1f}")
                    print("   IN:")
                    for p_in in rec['in']:
                        print(f"     - {p_in.web_name} (Buying at {p_in.cost_str}) - AdjScore {get_display_score(p_in):.1f}")
            else:
                # Fallback: show best available if no matching FT count
                print(f"\nNo {free_transfers}-transfer options found. Best available:")
                for i, rec in enumerate(recommendations[:3]):
                    ft_cost = len(rec['out'])
                    print(f"{i+1}. {transfer_names.get(ft_cost, str(ft_cost)+'-player')} Transfer (Delta: {rec['delta']:.1f}, FT cost: {ft_cost}):")
                    print("   OUT:")
                    for p_out in rec['out']:
                        print(f"     - {p_out.web_name} (Selling at {p_out.selling_cost_str}) - AdjScore {get_display_score(p_out):.1f}")
                    print("   IN:")
                    for p_in in rec['in']:
                        print(f"     - {p_in.web_name} (Buying at {p_in.cost_str}) - AdjScore {get_display_score(p_in):.1f}")
            
            # Roll FT suggestion
            if should_roll:
                print(f"\n💡 Consider ROLLING your FT (delta/transfer = {delta_per_ft:.1f} < {ROLL_THRESHOLD}).")
                print("   Low improvement suggests saving FT for a better opportunity.")
            elif free_transfers == 1 and best_single_delta < ROLL_THRESHOLD:
                print(f"\n💡 Consider ROLLING your FT to have 2 next week (best single delta = {best_single_delta:.1f}).")
        return

    # Default behavior: Select best squad
    squad = select_best_squad(players)

    print("Best ICT x PPG Squad (Position, Fixture, Form, Count, Minutes Adjusted)")
    for position, players_in_pos in squad.items():
        # Re-sort by adjusted score with tripled fixture boost for consistency
        sorted_players = sorted(
            players_in_pos,
            key=get_display_score,
            reverse=True
        )
        print_position_block(position, sorted_players)


if __name__ == "__main__":
    main()
