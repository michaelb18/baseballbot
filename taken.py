import pandas as pd

def get_taken():
    taken = pd.read_csv('./RosterGrid.csv')
    rostered = []
    for position in taken.keys():
        unavailable = taken[position].dropna().to_numpy()
        for player in unavailable:
            for p in player.split('\n'):
                rostered.append(p)
    return rostered

print(get_taken())
assert 'B. Rodgers' in get_taken()