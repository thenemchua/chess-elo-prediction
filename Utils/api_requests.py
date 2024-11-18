import requests

def get_player_stats(username, headers):
    url = f"https://api.chess.com/pub/player/{username}/stats"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return response.status_code


def get_country_players(country_code, headers):
    '''
    country_code uses iso 3166 country codes
    returns a list of players' username
    returns an empty list if no players in the country
    '''
    url = f"https://api.chess.com/pub/country/{country_code}/players"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['players']
    return []


def get_player_games(username, YYYY, MM, headers):
    url = f"https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}"
    response = requests.get(url,headers=headers)
    if response.status_code == 200:
        return response.json()['games']
    return []


def get_titled_player_usernames(title_abbrev, headers):
    '''
    Possible titles are: GM, WGM, IM, WIM, FM, WFM, NM, WNM, CM, WCM
    '''
    url = f"https://api.chess.com/pub/titled/{title_abbrev}"
    response = requests.get(url,headers=headers)
    if response.status_code == 200:
        return response.json()['players']
    return []