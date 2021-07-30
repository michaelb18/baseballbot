import pandas as pd
import numpy as np
def get_good_for_stat(players, categories=['HR'], reversals = [False], quantiles = [.70,.97]):
    lower, upper = players[categories[0]].quantile(quantiles)
    p = players.loc[(players[categories[0]] >= lower) & (players[categories[0]] <= upper)]['Player'].to_numpy()
    print(str(categories[0]) + ' ' + str(lower) + ' ' + str(upper))
    i = 1
    for category in categories[1:]:
        lower, upper = players[category].quantile(quantiles)
        if(reversals[i]):
            lower, upper = players[category].quantile([1-quantiles[1], 1-quantiles[0]])
        print(str(category) + ' ' + str(lower) + ' ' + str(upper))
        p = np.intersect1d(p, players.loc[(players[category] >= lower) & (players[category] <= upper)]['Player'].to_numpy())
        i = i + 1
    return p
    
    #return s
    
players = pd.read_csv('FantasyPros_2021_Projections_H.csv').dropna(how = 'any')
pitchers = pd.read_csv('FantasyPros_2021_Projections_P.csv').dropna(how = 'any')
pitchers = pitchers.loc[pitchers['Positions'] == 'SP']
#start of season percentiles:
#play = get_good_for_stat(players, categories = ['HR', 'RBI', 'R', 'AVG'], reversals = [False, False, False, False], quantiles = [.70,.97])
play = get_good_for_stat(players, categories = ['HR', 'RBI', 'R', 'AVG'], reversals = [False, False, False, False], quantiles = [.7,.97])

players.loc[players['Player'].isin(play)].to_csv('batters.csv')


#start of season percentiles:
pitch = get_good_for_stat(pitchers, categories = ['K', 'W', 'ERA', 'WHIP'], reversals = [False, False, True, True], quantiles = [.75, 1])
#pitch = get_good_for_stat(pitchers, categories = ['ERA', 'WHIP'], reversals = [True, False, True], quantiles = [0, .1])
pitchers.loc[pitchers['Player'].isin(pitch)].to_csv('pitchers.csv')