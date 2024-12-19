import pandas as pd
import re
import datetime
import numpy as np
from Utils import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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


def extract_moves_chess(pgn):
    # Extract the moves section (after metadata)
    moves_section = re.search(r"1\..*", pgn).group()

    # Remove move numbers (e.g., "1.", "2.", etc.)
    cleaned_moves = re.sub(r"\d+\.", "", moves_section)

    # Split into a list of moves
    moves = cleaned_moves.split()

    # Remove non-move entries (e.g., result {1-0}, {0-1}, {1/2-1/2})
    filtered_moves = [move for move in moves if not re.match(r"[{}]", move)]

    # Join the moves back into a single string
    result = " ".join(filtered_moves).strip()

    # Remove any remaining '{' or '}' characters in the final string
    result = result.replace("{", "").replace("}", "")

    return result


# Preprocessing pour baseline

def create_X_from_initial_data_for_baseline(df):
    """
    Crée X à partir de la donnée de base.
    X: le pgn global en format string, chaque coup séparé par un espace vide.

    """
    df=extract_moves_and_times_pgn(df)
    X = df[["pgn_all"]]
    X= X["pgn_all"].apply(lambda x: " ".join(x))
    return X

def create_y_from_initial_data_for_baseline(df):
    """
    Crée y à partir de la donnée de base.
    y: le rating de white.
    """
    y=df[["white_rating"]]
    return y

def tokeniser_pgn(x, max_features):
    """
    Permet de tokeniser x, x étant en entrée un df contenant les pgn en format str avec un espace vide entre chaque coup joué.
    """

    tk = Tokenizer(num_words=max_features, filters=".,",oov_token=-1)
    tk.fit_on_texts(x)
    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')
    X_token = tk.texts_to_sequences(x)
    return X_token, tk

def pad_sequence_X(x_token, maxlen):
    """
    Transforme tous les X pour avoir la même taille
    """
    X = sequence.pad_sequences(x_token, maxlen=maxlen)
    return X

def max_len_baseline(X):
    all_list_pgn=[]
    for pgn in X["pgn_all"]:
        all_list_pgn.append(len(pgn))
    return max(all_list_pgn)

def max_features_baseline(X):
    all_pgn=[]
    for pgn in X["pgn_all"]:
        for ele in pgn.split(" "):
            all_pgn.append(ele)
    return len(set(all_pgn))

def preprocessing_baseline_francois(X,tk=None):
    """
    X input = df incluant uniquement X["pgn_all"]
    """
    # max_len=max_len_baseline(pd.DataFrame(X))
    max_features=max_features_baseline(pd.DataFrame(X)) #changement
    print(f'max_features: {max_features}')
    if tk:
        X=tk.texts_to_sequences(X)
        print("Use tk already fitted")
    else:
        X,tk=tokeniser_pgn(X,max_features)
    X=pad_sequence_X(X, 250)
    return X,tk


def prepro_df_to_model_baseline_jules(df):

    df["pgn_all"]=df["pgn"]
    X = df["pgn_all"]
    y=create_y_from_initial_data_for_baseline(df)

    return X,y



****
*****
DELETE?

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
