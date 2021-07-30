import pandas as pd
import numpy as np

pitching_categories = ['K', 'W', 'ERA', 'WHIP']
batting_categories = ['HR', 'RBI', 'R', 'AVG'] 

players = pd.read_csv('FantasyPros_2021_Projections_H.csv').dropna(how = 'any')
pitchers = pd.read_csv('FantasyPros_2021_Projections_P.csv').dropna(how = 'any')
list_players = pd.read_csv('batters.csv').dropna(how = 'any')
list_pitchers = pd.read_csv('pitchers.csv').dropna(how = 'any')
drafted_players = pd.DataFrame(columns = batting_categories)
drafted_pitchers = pd.DataFrame(columns = pitching_categories)

def df_len(dF):
    return len(dF.index)
def project_stats(drafted_players, drafted_pitchers):
    total_ab = drafted_players['AB'].sum()
    total_h = drafted_players['H'].sum()
    total_hr = drafted_players['HR'].sum()
    total_rbi = drafted_players['RBI'].sum()
    total_r = drafted_players['R'].sum()
    total_ba = total_h/total_ab

    total_hits = drafted_pitchers['H'].sum()
    total_er = drafted_pitchers['ER'].sum()
    total_inn = drafted_pitchers['IP'].sum()
    total_walks = drafted_pitchers['BB'].sum()
    total_k = drafted_pitchers['K'].sum()
    total_w = drafted_pitchers['W'].sum()
    total_whip = (total_hits + total_walks)/total_inn
    total_era = 9 * total_er/total_inn

    return [total_hr, total_rbi, total_r, total_ba, total_k, total_w, total_era, total_whip, total_h, total_hits, total_er, total_inn, total_walks, total_ab]

def suggest_players(drafted_players, drafted_pitchers, available_pitchers, available_batters):
    #iterating stats are ba, era, whip

    desired_avg = .265
    desired_homeruns = 300
    desired_rbi = 1200
    desired_r = 1200

    desired_ks = 1500
    desired_era = 3.50
    desired_whip = 1.15
    desired_w = 90

    hr, rbi, r, ba, k, w, era, whip, h, hits, er, inn, walks, ab = project_stats(drafted_players, drafted_pitchers)
    total_batters = df_len(drafted_players)
    total_pitchers = df_len(drafted_pitchers)
    batters_left = 15 - total_batters
    pitchers_left = 15 - total_pitchers

    hr_remaining_avg = (desired_homeruns - hr)/batters_left
    rbi_remaining_avg = (desired_rbi - rbi)/batters_left
    r_remaining_avg = (desired_r - r)/batters_left

    k_remaining_avg = (desired_ks - k)/pitchers_left
    w_remianing_avg = (desired_w - w)/pitchers_left

    all_pitchers = {}
    inverse_all_pitchers = {}
    for pitcher in available_pitchers:
        w_difference = (pitcher['W'] - w_remianing_avg)/pitcher['W']
        k_difference = (pitcher['K'] - k_remianing_avg)/pitcher['K']
        
        projected_er = er + pitcher['ER']
        projected_inn = inn + pitcher['IP']
        projected_walks = walks + pitcher['BB']
        projected_hits = hits + pitcher['H']
        projected_era = projected_er/projected_inn * 9
        projected_whip = (projected_walks + projected_hits)/projected_inn

        era_difference = (projected_era - desired_era)/projected_era
        whip_difference = (projected_whip - desired_whip)/projected_whip

        total_difference = w_difference + k_difference + era_difference + whip_difference
        all_pitchers[pitcher] = total_difference
        inverse_all_pitchers[total_difference] = pitcher
    all_pitchers_scores = sorted(all_pitchers.values())

    top_pitchers = [inverse_all_pitchers[all_pitchers_scores[0]], inverse_all_pitchers[all_pitchers_scores[1]], inverse_all_pitchers[all_pitchers_scores[2]]]
    all_batters = {}
    inverse_all_batters = {}
    for batter in available_batters:
        hr_difference = (batter['HR'] - w_remianing_avg)/batter['HR']
        rbi_difference = (batter['RBI'] - k_remianing_avg)/batter['RBI']
        r_difference = (batter['R'] - r_remaining_avg)/batter['R']

        projected_ba = (h + batter['H'])/(ab + batter['AB'])
        ba_difference = (projected_ba - desired_avg)/projected_ba
        total_difference = hr_difference + rbi_difference + ba_difference + r_difference
        all_batters[pitcher] = total_difference
        inverse_all_batters[total_difference] = batter
    all_batter_scores = sorted(all_batters.values())

    top_batters = [inverse_all_batters[all_batters_scores[0]], inverse_all_batters[all_batters_scores[1]], inverse_all_batters[all_batters_scores[2]]]
def show_team(drafted_players, drafted_pitchers):
    print('Batters')
    print('-' * 50)
    print(drafted_players)
    print('Pitchers')
    print('-' * 50)
    print(drafted_pitchers)

def draft_pitcher(pitcher, drafted_pitchers, pitchers, list_pitchers, my_team = True):
    if my_team:
        drafted_pitchers = drafted_pitchers.append(pitcher)
    ind = pitchers[pitchers['Player'] == pitcher['Player'].values[0]].index
    if len(ind.values) > 0:
        pitchers.drop(ind, inplace=True)

    list_ind = list_pitchers[list_pitchers['Player'] == pitcher['Player'].values[0]].index
    if len(list_ind.values) > 0:
        list_pitchers.drop(list_ind, inplace=True)
    return drafted_pitchers

def draft_player(player, drafted_players, players, list_players, my_team = True):
    if my_team:
        drafted_players = drafted_players.append(player)
    ind = players[players['Player'] == player['Player'].values[0]].index
    if len(ind.values) > 0:
        players.drop(ind, inplace=True)

    list_ind = list_players[list_players['Player'] == player['Player'].values[0]].index
    if len(list_ind.values) > 0:
        list_players.drop(list_ind, inplace=True)
    return drafted_players

def draft(player_name, players, pitchers, drafted_pitchers, list_pitchers, drafted_players, list_players, my_team = True):
    
    player = players.loc[players['Player'] == player_name]
    pitcher = pitchers.loc[pitchers['Player'] == player_name]
    
    if len(player.index.values) == 0 and len(pitcher.index.values) == 0:
        print('That player wasnt found')
        return (False, drafted_players, drafted_pitchers)
    if len(player.index.values) > 0:
        drafted_players = draft_player(player, drafted_players, players, list_players, my_team)
    elif len(pitcher.index.values) > 0:
        drafted_pitchers = draft_pitcher(pitcher, drafted_pitchers, pitchers, list_pitchers, my_team)
    return (True, drafted_players, drafted_pitchers)

spot = 0
rounds = 5#int(input('Enter number of rounds'))
teams = 5#int(input('Enter number of teams'))
my_spot = 5#int(input('Enter draft position'))
while spot < rounds * teams:
    succ, drafted_players, drafted_pitchers = draft(input('Enter player to draft: '), players, pitchers, drafted_pitchers, list_pitchers, drafted_players, list_players, spot % teams == my_spot)
    while not succ :
        succ, drafted_players, drafted_pitchers = draft(input('Enter player to draft: '), players, pitchers, drafted_pitchers, list_pitchers, drafted_players, list_players, spot % teams == my_spot)
    show_team(drafted_players, drafted_pitchers)
    spot = spot + 1