import pandas as pd
import numpy as np
from taken import get_taken
def get_good_for_stat(players, categories=['HR'], reversals = [False], quantiles = [[.70,.97]]*5):
    print(quantiles)
    players_taken = get_taken()
    lower, upper = players[categories[0]].quantile(quantiles[0])
    p = players.loc[(players[categories[0]] >= lower) & (players[categories[0]] <= upper)]['Player'].to_numpy()
    
    print(str(categories[0]) + ' ' + str(lower) + ' ' + str(upper))
    i = 1
    for category in categories[1:]:
        lower, upper = players[category].quantile(quantiles[i])
        if(reversals[i]):
            lower, upper = players[category].quantile([1-quantiles[i][1], 1-quantiles[i][0]])
        print(str(category) + ' ' + str(lower) + ' ' + str(upper))
        p = np.intersect1d(p, players.loc[(players[category] >= lower) & (players[category] <= upper)]['Player'].to_numpy())
        i = i + 1
    r = 0
    p2 = np.array([])
    for person in p:
        name = person.split(' ')
        checkstr = name[0][0]+'. '+name[1]
        if checkstr not in players_taken:
            p2 = np.append(p2, person)
        r = r + 1
    return p2
    
    #return s
    
players = pd.read_csv('FantasyPros_2022_Projections_H.csv').dropna(how = 'any')
pitchers = pd.read_csv('FantasyPros_2022_Projections_P.csv').dropna(how = 'any')
pitchers = pitchers.loc[pitchers['Positions'] == 'SP']
#start of season percentiles:
#play = get_good_for_stat(players, categories = ['HR', 'RBI', 'R', 'AVG'], reversals = [False, False, False, False], quantiles = [.70,.97])

lower_bound = .95
play = get_good_for_stat(players, categories = ['HR', 'RBI', 'R', 'AVG'], reversals = [False, False, False, False], quantiles = [[.6,.97],[.7,.97],[.70,.97],[.60,.97]])
#while len(play) < 35:
#    lower_bound = lower_bound - .01
#    play = get_good_for_stat(players, categories = ['HR', 'RBI', 'R', 'AVG'], reversals = [False, False, False, False], quantiles = [.70,.97])
#    print('-'*50)

players.loc[players['Player'].isin(play)].sample(frac=1).reset_index().drop(columns=['index']).to_csv('batters.csv', index = False)

lower_bound = 1
#start of season percentiles:
pitch = get_good_for_stat(pitchers, categories = ['K', 'W', 'ERA', 'WHIP'], reversals = [False, False, True, True], quantiles = [[.75, 1], [0, 1], [.77, 1], [.75, 1]])
#while len(pitch) < 50:
#    lower_bound = lower_bound - .05
#    print(lower_bound)
#    pitch = get_good_for_stat(pitchers, categories = ['K', 'W', 'ERA', 'WHIP'], reversals = [False, False, True, True], quantiles = [lower_bound, 1])
#    print('-'*50)
pitchers.loc[pitchers['Player'].isin(pitch)].sample(frac=1).reset_index().drop(columns=['index']).to_csv('pitchers.csv', index = False)