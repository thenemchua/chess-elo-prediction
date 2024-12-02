import pandas as pd
import re
import datetime
import numpy as np
from Utils import utils

def extract_opening(df):
    df["opening"] = df["opening"].str.extract(r'openings/([^\.]+)')
    return df


def func_time_control(pgn):
    """
    Récupère le time control dans le PGN - durée de la partie maximale
    ex: 180s, 300s
    """

    time_control = re.findall(r'\[TimeControl "([^"]+)"\]', pgn)[0] # Utilisation d'une expression régulière pour extraire TimeControl
    increment=0
    for char in time_control:
        if char=="+":
            increment=float(time_control.split("+")[1])
            time_control=float(time_control.split("+")[0])
    return int(time_control)


def func_increment(pgn):
    """
    Récupère l'increment dans le PGN - durée ajoutée au temps à jouer après chaque coup.
    ex: 0s, 2s, 5s
    """

    time_control = re.findall(r'\[TimeControl "([^"]+)"\]', pgn)[0] # Utilisation d'une expression régulière pour extraire TimeControl
    increment=0
    for char in time_control:
        if char=="+":
            increment=float(time_control.split("+")[1])
            time_control=float(time_control.split("+")[0])
    return int(increment)


def times_per_color(pgn):
    """
    Permet de récupérer 3 listes de temps par coups joués: all, white, black
    ex: 1.2s pour le premier coup joué etc.
    """

    time_pattern = r"\{\[%clk ([0-9:.]+)\]\}" # Utilisation d'une expression régulière pour extraire le compteur de temps restant à chaque coup joué
    times = re.findall(time_pattern, pgn)

    time_control = func_time_control(pgn) #compteur de départ pour chaque joueur
    increment = func_increment(pgn) #increment à ajouter pour chaque coup

    duration_w=time_control
    duration_b=time_control

    times_w=[]
    times_b=[]
    times_all=[]

    i=0

    for time in times:
        if i%2==0:
            h,m,s = time.split(':') # split la donnée str en 3 valeurs
            seconds=float(datetime.timedelta(hours=float(h),minutes=float(m),seconds=float(s)).total_seconds())  # calcul le total de secondes restantes
            move_seconds=duration_w-seconds+increment # temps écoulé pour jouer ce coup
            new_time=duration_w-seconds # temps restant après ce coup
            times_w.append(new_time)
            duration_w=seconds
        else:
            h,m,s = time.split(':')
            seconds=float(datetime.timedelta(hours=float(h),minutes=float(m),seconds=float(s)).total_seconds())
            move_seconds=duration_b-seconds+increment
            new_time=duration_b-seconds
            times_b.append(new_time)
            duration_b=seconds
        i+=1
        times_all.append(new_time)

    return times_all,times_w, times_b


def pgn_per_color(pgn):
    """
    Permet de récupérer 3 listes pour les coups joués: all, white, black
    """

    move_pattern = r"([0-9]+\.\s+\S+|\.\.\.\s+\S+)" # regex pour les coups joués par les joueurs
    moves = re.findall(move_pattern, pgn)
    moves = [move.split()[-1] for move in moves]  # On garde uniquement le coup, pas le numéro.

    moves_w=[]
    moves_b=[]
    i=0

    for move in moves:
        if i%2==0: # 1 coup sur 2
            moves_w.append(move)
        else:
            moves_b.append(move)
        i+=1

    return moves,moves_w, moves_b

def extract_moves_and_times_pgn(df):
    """
    Input = df de base venant du fichier .json
    Permet de créer les colonnes time_control, increment, pgn_all, pgn_w, pgn_b, times_all, times_w, times_b
    """
    df["time_control"]=df["pgn"].apply(lambda x: func_time_control(x))
    df["increment"]=df["pgn"].apply(lambda x: func_increment(x))
    df["pgn_all"]=df["pgn"].apply(lambda x: pgn_per_color(x)[0])
    df["pgn_w"]=df["pgn"].apply(lambda x: pgn_per_color(x)[1])
    df["pgn_b"]=df["pgn"].apply(lambda x: pgn_per_color(x)[2])
    df["times_all"]=df["pgn"].apply(lambda x: times_per_color(x)[0])
    df["times_w"]=df["pgn"].apply(lambda x: times_per_color(x)[1])
    df["times_b"]=df["pgn"].apply(lambda x: times_per_color(x)[2])
    df["opening"] = df["opening"].str.extract(r'openings/([^\.]+)')
    return df


def extract_moves_and_times_pgn_2(df):
    """
    Input = df de base venant du fichier .json
    Permet de créer les colonnes time_control, increment, pgn_all, pgn_w, pgn_b, times_all, times_w, times_b
    """
    df["time_control"]=df["pgn"].apply(lambda x: func_time_control(x))
    df["increment"]=df["pgn"].apply(lambda x: func_increment(x))
    df["pgn_all"]=df["pgn"].apply(lambda x: " ".join(pgn_per_color(x)[0]))
    df["pgn_w"]=df["pgn"].apply(lambda x: " ".join(pgn_per_color(x)[1]))
    df["pgn_b"]=df["pgn"].apply(lambda x: " ".join(pgn_per_color(x)[2]))
    df["times_all"]=df["pgn"].apply(lambda x: times_per_color(x)[0])
    df["times_w"]=df["pgn"].apply(lambda x: times_per_color(x)[1])
    df["times_b"]=df["pgn"].apply(lambda x: times_per_color(x)[2])
    df["opening"] = df["opening"].str.extract(r'openings/([^\.]+)')
    return df

def pgn_from_chess_com(pgn):
    """
    Input = pgn provenant de chess.com
    Permet de préprocesser le pgn provenant directement de chess.com
    """
    # Extraire les lignes de mouvements et les nettoyer
    moves = ' '.join(line for line in pgn.split('\n') if not line.startswith('[')).strip()
    # Supprimer les numéros de coups (ex : "1.", "2.")
    moves = re.sub(r'\d+\.\s*', '', moves)
    # Supprimer le résultat à la fin (0-1, 1-0, ou 1/2-1/2), s'il existe
    moves = re.sub(r'\s?(0-1|1-0|1/2-1/2)\s?$', '', moves[:-1])
    return moves.strip()

def pgn_from_lichess(pgn):
    """
    Input = pgn provenant de lichess
    Permet de préprocesser le pgn brute provenant directement de lichess
    """
    # Extraire les lignes de mouvements et les nettoyer
    moves = ' '.join(line for line in pgn.split('\n') if not line.startswith('[')).strip()
    # Supprimer les numéros de coups (ex : "1.", "2.")
    moves = re.sub(r'\d+\.\s*', '', moves)
    # # Supprimer le résultat à la fin (0-1, 1-0, ou 1/2-1/2), s'il existe
    moves = re.sub(r'\s?(0-1|1-0|1/2-1/2)\s?$', '', moves)
    return moves.strip()
