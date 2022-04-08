import pandas as pd
import numpy as np
import re
#keeping price url so I can export in the future
prices_url = 'https://www.fangraphs.com/auctiontool.aspx?type=bat&proj=steamer&pos=1,1,1,1,4,1,0,1,0,5,7,2,0,0,0&dollars=260&teams=15&mp=5&msp=5&mrp=5&mb=1&split=&points=c|0,1,2,3,4|0,1,2,3,4&lg=MLB&rep=0&drp=0&pp=C,SS,2B,3B,OF,1B&players='
pitching_categories = ['K', 'W', 'ERA', 'WHIP', 'Player', 'Team', 'Positions', 'IP', 'SV', 'ER', 'H', 'BB', 'HR', 'G', 'GS', 'L', 'CG']
batting_categories = ['HR', 'RBI', 'R', 'AVG', 'Player', 'Team', 'Positions', 'AB', 'SB', 'OBP', 'H', '2B', '3B', 'BB', 'SO', 'SLG', 'OPS']  

players = pd.read_csv('FantasyPros_2022_Projections_H.csv').dropna(how = 'any')
pitchers = pd.read_csv('FantasyPros_2022_Projections_P.csv').dropna(how = 'any')
list_players = pd.read_csv('batters.csv').dropna(how = 'any')
list_pitchers = pd.read_csv('pitchers.csv').dropna(how = 'any')
drafted_players = pd.DataFrame(columns = batting_categories)
drafted_pitchers = pd.DataFrame(columns = pitching_categories)
batters_dollars = pd.read_csv('FanGraphs Leaderboard Batters.csv').dropna(how = 'any')
pitchers_dollars = pd.read_csv('FanGraphs Leaderboard Pitchers.csv').dropna(how = 'any')
price_batters = {}
list_price_batters = {}
batters_dollars['PlayerName'] = [re.sub("\.", "",batter) for batter in batters_dollars['PlayerName']]
pitchers_dollars['PlayerName'] = [re.sub("\.", "",pitcher) for pitcher in pitchers_dollars['PlayerName']]
#turn dollars into numeric and append column-wise to players and pitchers, add to list_players and list_pitchers if applicable.
for batter in players.iterrows():
    name = re.sub("\.", "",batter[1]['Player'])
    if not batters_dollars[batters_dollars['PlayerName'] == name].empty:
        price = re.sub("[^0-9^.]", "", batters_dollars[batters_dollars['PlayerName'] == name]['Dollars'].to_numpy()[0][1:])
        price_batters[batter[1]['Player']] = float(price)
        if batter[1]['Player'] in list_players['Player']:
            list_price_batters[batter[1]['Player']] = float(price)
    else:
        price_batters[batter[1]['Player']] = 1.0
        if batter[1]['Player'] in list_players['Player']:
            list_price_batters[batter[1]['Player']] = 1.0
#print(list_price_batters)
#print(price_batters)

price_pitchers = {}
list_price_pitchers = {}
for pitcher in pitchers.iterrows():
    name = re.sub("\.", "",pitcher[1]['Player'])
    if not pitchers_dollars[pitchers_dollars['PlayerName'] == name].empty:
        price = re.sub("[^0-9^.]", "", pitchers_dollars[pitchers_dollars['PlayerName'] == name]['Dollars'].to_numpy()[0][1:])
        price_pitchers[pitcher[1]['Player']] = float(price)
        if pitcher[1]['Player'] in list_pitchers['Player']:
            list_price_pitchers[pitcher[1]['Player']] = float(price)
    else:
        price_pitchers[pitcher[1]['Player']] = 1.0
        if pitcher[1]['Player'] in list_pitchers['Player']:
            list_price_pitchers[pitcher[1]['Player']] = 1.0

#print(list_pitchers)
#print(list_players)
#print(players)
#print(pitchers)
budget = 260
batter_budget = budget/2.7
pitcher_budget = budget - batter_budget
assert pitcher_budget == 1.7 * batter_budget
def determine_capital_availability(possible_players, possible_pitchers, batters_dollars, pitchers_dollars, pitcher_dollars_available, batter_dollars_available):
    #linear damp
    max_spending_pitcher = .1 * pitcher_dollars_available
    max_spending_batter = .1 * batter_dollars_available
    return (possible_players[possible_players["Dollars"] <= max_spending_batter], possible_pitchers[possible_pitchers["Dollars"] <= max_spending_pitcher])
def df_len(dF):
    return len(dF.index)
def project_stats(drafted_players, drafted_pitchers):
    total_ab = drafted_players['AB'].sum()
    total_h = drafted_players['H'].sum()
    total_hr = drafted_players['HR'].sum()
    total_rbi = drafted_players['RBI'].sum()
    total_r = drafted_players['R'].sum()
    
    total_ba = 0
    if total_ab != 0:
        total_ba = total_h/total_ab
    total_hits = drafted_pitchers['H'].sum()
    total_er = drafted_pitchers['ER'].sum()
    total_inn = drafted_pitchers['IP'].sum()
    total_walks = drafted_pitchers['BB'].sum()
    total_k = drafted_pitchers['K'].sum()
    total_w = drafted_pitchers['W'].sum()
    total_whip = 0
    total_era = 0
    if total_inn != 0:
        total_whip = (total_hits + total_walks)/total_inn
        total_era = 9 * total_er/total_inn

    return [total_hr, total_rbi, total_r, total_ba, total_k, total_w, total_era, total_whip, total_h, total_hits, total_er, total_inn, total_walks, total_ab]

def suggest_players(drafted_players, drafted_pitchers, available_pitchers, available_batters, averages = False):
    #iterating stats are ba, era, whip

    desired_avg = .265
    desired_homeruns = 300
    desired_rbi = 1200
    desired_r = 1200

    desired_ks = 1350
    desired_era = 3.65
    desired_whip = 1.15
    desired_w = 75

    hr, rbi, r, ba, k, w, era, whip, h, hits, er, inn, walks, ab = project_stats(drafted_players, drafted_pitchers)
    total_batters = df_len(drafted_players)
    total_pitchers = df_len(drafted_pitchers)
    batters_left = 15 - total_batters
    pitchers_left = 7 - total_pitchers
    if batters_left > 0:
        hr_remaining_avg = (desired_homeruns - hr)/batters_left
        rbi_remaining_avg = (desired_rbi - rbi)/batters_left
        r_remaining_avg = (desired_r - r)/batters_left

    top_pitchers = []
    if pitchers_left > 0:
        k_remaining_avg = (desired_ks - k)/pitchers_left
        w_remianing_avg = (desired_w - w)/pitchers_left
        all_pitchers = {}
        inverse_all_pitchers = {}
        for index, pitcher in available_pitchers.iterrows():
            w_difference = (pitcher['W'] - w_remianing_avg)/pitcher['W']
            k_difference = (pitcher['K'] - k_remaining_avg)/pitcher['K']
            
            projected_er = er + pitcher['ER']
            projected_inn = inn + pitcher['IP']
            projected_walks = walks + pitcher['BB']
            projected_hits = hits + pitcher['H']
            projected_era = projected_er/projected_inn * 9
            projected_whip = (projected_walks + projected_hits)/projected_inn

            era_difference = (projected_era - desired_era)/projected_era
            whip_difference = (projected_whip - desired_whip)/projected_whip
            total_difference = abs(w_difference) + abs(k_difference) + abs(era_difference) + abs(whip_difference)
            print(pitcher['Player'])
            print(total_difference)
            all_pitchers[pitcher['Player']] = abs(total_difference)
            inverse_all_pitchers[abs(total_difference)] = pitcher
        all_pitchers_scores = sorted(all_pitchers.values())
        #all_pitchers_scores.reverse()
        if averages:
            print('Drafed based on averages:')
            print('W:'+str(w_remianing_avg))
            print('K:'+str(k_remaining_avg))
            print('ERA'+str((desired_era - era)/pitchers_left))
            print('WHIP'+str((desired_whip - whip)/pitchers_left))
        with open('top_pitchers.csv', 'w') as arms:
            for arm in range(len(all_pitchers_scores)):
                arms.write(str(inverse_all_pitchers[all_pitchers_scores[arm]]['Player']) + ':' + str(all_pitchers_scores[arm]) + '\n')
        top_pitchers = [inverse_all_pitchers[all_pitchers_scores[0]], inverse_all_pitchers[all_pitchers_scores[1]], inverse_all_pitchers[all_pitchers_scores[2]]]
    top_batters = []
    if batters_left > 0:
        all_batters = {}
        inverse_all_batters = {}
        for index, batter in available_batters.iterrows():
            hr_difference = (batter['HR'] - hr_remaining_avg)/batter['HR']
            rbi_difference = (batter['RBI'] - rbi_remaining_avg)/batter['RBI']
            r_difference = (batter['R'] - r_remaining_avg)/batter['R']
            projected_ba = (h + batter['H'])/(ab + batter['AB'])
            ba_difference = (projected_ba - desired_avg)/projected_ba
            total_difference = hr_difference + rbi_difference + ba_difference + r_difference
            all_batters[batter['Player']] = abs(total_difference)
            inverse_all_batters[abs(total_difference)] = batter
        all_batters_scores = sorted(all_batters.values())
        #all_batters_scores.reverse()
        with open('top_batters.csv', 'w') as bats:
            for bat in range(len(all_batters_scores)):
                bats.write(str(inverse_all_batters[all_batters_scores[bat]]['Player']) + ':' + str(all_batters_scores[bat]) + '\n')
        top_batters = [inverse_all_batters[all_batters_scores[0]], inverse_all_batters[all_batters_scores[1]], inverse_all_batters[all_batters_scores[2]]]

    return (top_batters, top_pitchers)
def show_team(drafted_players, drafted_pitchers):
    print('Batters')
    print('-' * 50)
    print(drafted_players)
    print('Pitchers')
    print('-' * 50)
    print(drafted_pitchers)
    projection = project_stats(drafted_players, drafted_pitchers)
    print('Total BA: ' + str(projection[3]))
    print('Total HR: ' + str(projection[0]))
    print('Total RBI: ' + str(projection[1]))
    print('Total R: ' + str(projection[2]))
    print('-' * 50)
    print('Total K: ' + str(projection[4]))
    print('Total W: ' + str(projection[5]))
    print('Total ERA: ' + str(projection[6]))
    print('Total WHIP: ' + str(projection[7]))
def save_draft(drafted_players, drafted_pitchers, available_players, available_pitchers):
    drafted_players.to_csv('drafted_players.csv', index=False)
    drafted_pitchers.to_csv('drafted_pitchers.csv', index=False)
    available_players.to_csv('available_players.csv', index=False)
    available_pitchers.to_csv('available_pitchers.csv', index=False)
def load_draft():
    drafted_players = pd.read_csv('drafted_players.csv')
    drafted_pitchers = pd.read_csv('drafted_pitchers.csv')
    available_players = pd.read_csv('available_players.csv')
    available_pitchers = pd.to_csv('available_pitchers.csv')

    return (drafted_players, drafted_pitchers, available_players, available_pitchers)
def draft_pitcher(pitcher, drafted_pitchers, pitchers, list_pitchers, my_team = True):
    if my_team:
        if df_len(drafted_pitchers) >= 7:
            print('Cant draft another pitcher!')
            return drafted_pitchers
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
        if df_len(drafted_players) >= 15:
            print('Cant draft another batter!')
            return drafted_players
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
my_spot = 1#int(input('Enter draft position'))
while spot < rounds * teams:
    cmd = input('michael w/ LDT >')
    cmd_split = cmd.split()
    #print(cmd_split[0] == 'draft_me' or cmd_split[0] == 'dm')
    #print(cmd_split)
    #print(cmd_split[0])
    if cmd_split[0] == 'draft':
        team_name_index = cmd.find('--team=\"') + len('--team=\"')
        team_name_end = cmd.find('\"', team_name_index)
        team_name = cmd[team_name_index:team_name_end]

        player_name_index = cmd.find('--player=\"') + len('--player=\"')
        player_name_end = cmd.find('\"', player_name_index)
        player_name = cmd[player_name_index:player_name_end]  
        succ, drafted_players, drafted_pitchers = draft(player_name, players, pitchers, drafted_pitchers, list_pitchers, drafted_players, list_players, team_name == 'Bald Eagles')
        if succ:
            spot = spot + 1
    elif cmd_split[0] == 'draft_me' or cmd_split[0] == 'dm':
        player_name = cmd_split[1]
        for c in cmd_split[2:]:
            player_name = player_name + ' ' + c 
        succ, drafted_players, drafted_pitchers = draft(player_name, players, pitchers, drafted_pitchers, list_pitchers, drafted_players, list_players, True)
        if succ:
            spot = spot + 1
    elif cmd_split[0] == 'draft_other' or cmd_split[0] == 'do':
        player_name = cmd_split[1]
        for c in cmd_split[2:]:
            player_name = player_name + ' ' + c 
        succ, drafted_players, drafted_pitchers = draft(player_name, players, pitchers, drafted_pitchers, list_pitchers, drafted_players, list_players, False)
        if succ:
            spot = spot + 1
    elif cmd_split[0] == 'show_team':
        show_team(drafted_players, drafted_pitchers)
    elif cmd_split[0] == 'suggest_players':
        print(suggest_players(drafted_players, drafted_pitchers, list_pitchers, list_players))
    else:
        print('not a command')