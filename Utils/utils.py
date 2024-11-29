import os
import sys

import multiprocessing

import csv
import json
import ijson

from collections import defaultdict
import re

import pandas as pd
import numpy as np

from google.cloud import storage
from Utils import preprocessing

from io import BytesIO


def create_game_dict():
    '''
    Crée le format de data qu'on veut exploiter
    '''
    return {'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]}


def create_evaluated_game_dict():
    '''
    Crée un dictionnaire avec les colonnes après évaluation
    '''
    return {'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[], 'evaluation':[], 'best_move':[], 'mate':[]}


def create_games_dict():
    '''
    Crée un dictionnaire vide de données par mode de jeu
    '''
    games = {
        'bullet':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'blitz':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'rapid':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'daily':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]}
    }
    
    return games


def fill_games_dict(game_list, games):
    '''
    games parameter must be in this format:
    games = {
        'bullet':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'blitz':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'rapid':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'daily':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]}
    }
    '''
    rules = 'chess'
    rated = True
    
    for game in game_list:
        if game['rules'] == rules and game['rated'] == rated:
            games[game['time_class']]['fen'].append(game['fen'])
            games[game['time_class']]['pgn'].append(game['pgn'])
            games[game['time_class']]['white_rating'].append(game['white']['rating'])
            games[game['time_class']]['black_rating'].append(game['black']['rating'])
            games[game['time_class']]['white_result'].append(game['white']['result'])
            games[game['time_class']]['black_result'].append(game['black']['result'])
            games[game['time_class']]['opening'].append(game['eco'])


def save_list_to_csv(filename, l):
    '''
    filename is the path of the file
    l is the list you want to save as csv
    '''
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for e in l:
            writer.writerow([e])
    print(f'file saved to {filename}')


def read_csv_to_list(filename):
    '''
    filename is the path of the file
    returns a list with the values of the csv
    '''
    res = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            res.append(row[0])
            
    print(f'file loaded from {filename}')
    
    return res


def save_dict_to_json(filename, d):
    '''
    filename is the path of the file
    l is the list you want to save as json
    '''
    with open(filename, mode='w') as file:
        json.dump(d, file)
    print(f'file saved to {filename}')


def read_json_to_dict(filename):
    '''
    filename is the path of the file
    returns a list with the values of the json
    '''
    with open(filename, mode='r') as file:
        res = json.load(file)
        
    print(f'file loaded from {filename}')
    
    return res


def combine_games_to_dataframes(data_folder, file_prefix):
    """
    Combine les fichiers JSON contenant des parties d'échecs en plusieurs dataframes,
    un par contrôle de temps (bullet, blitz, rapid, daily).
    
    Args:
        data_folder (str): Chemin du dossier contenant les fichiers JSON.
        file_prefix (str): Préfixe des fichiers à inclure (e.g., "unknown_games_2024-10").
    
    Returns:
        dict: Un dictionnaire contenant un dataframe pour chaque contrôle de temps.
              Clés : 'bullet', 'blitz', 'rapid', 'daily'.
    """
    # Initialiser des dictionnaires pour chaque type de contrôle de temps
    games_dict = {
        'bullet': [],
        'blitz': [],
        'rapid': [],
        'daily': []
    }
    
    # Parcourir les fichiers dans le dossier data
    for file_name in os.listdir(data_folder):
        if file_name.startswith(file_prefix) and file_name.endswith(".json"):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Ajouter les données au dictionnaire correspondant
                for time_control in games_dict.keys():
                    if time_control in data:
                        games_dict[time_control].extend(data[time_control])
    
    # Convertir chaque liste de parties en dataframe
    dataframes = {}
    for time_control, games in games_dict.items():
        if games:
            df = pd.DataFrame(games)
            dataframes[time_control] = df
    
    return dataframes


def process_files_in_batches(data_folder, file_prefix, batch_size=10):
    files = [f for f in os.listdir(data_folder) if f.startswith(file_prefix) and f.endswith(".json")]
    
    # for i in range(0, len(files), batch_size):
    for i in range(0, 3, batch_size):
        batch_files = files[i:i+batch_size]
        batch_data = []
        
        for file in batch_files:
            filepath = os.path.join(data_folder, file)
            data = pd.read_json(filepath)
            batch_data.append(data)
        
        # Combine les fichiers d'un batch
        batch_df = pd.concat(batch_data, ignore_index=True)
        
        # Traite immédiatement pour libérer de la mémoire
        # process_dataframe(batch_df)  # Fonction de traitement à implémenter
        # del batch_data  # Libère la mémoire
    return batch_data
    print("Traitement terminé.")


def split_json_by_mode(input_file, output_dir, max_size_gb=0, num_parts=5):
    """
    Segmente un fichier JSON structuré par mode de jeu si sa taille dépasse une limite.
    
    Args:
        input_file (str): Chemin vers le fichier JSON à segmenter.
        output_dir (str): Répertoire où sauvegarder les fichiers segmentés.
        max_size_gb (float): Taille maximale autorisée pour un fichier JSON avant segmentation (en Go).
        num_parts (int): Nombre de segments à créer si le fichier est trop grand.
        
    Returns:
        None
    """
    # Vérifier la taille du fichier en Go
    file_size_gb = os.path.getsize(input_file) / (1024 ** 3)
    
    if file_size_gb > max_size_gb:
        print(f"Le fichier {input_file} fait {file_size_gb:.2f} Go et sera segmenté.")
        
        # Charger le JSON complet
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        game_modes = data.keys()  # ['bullet', 'blitz', 'rapid', 'daily']
        
        # Création du répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        for mode in game_modes:
            mode_data = data[mode]
            total_entries = len(mode_data['fen'])  # Chaque sous-liste doit avoir la même longueur
            chunk_size = total_entries // num_parts
            
            # Créer un sous-dossier pour chaque mode
            mode_output_dir = os.path.join(output_dir, mode)
            os.makedirs(mode_output_dir, exist_ok=True)
            
            # Segmenter les données pour chaque mode
            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_parts - 1 else total_entries
                
                chunk = {
                    key: value[start_idx:end_idx]
                    for key, value in mode_data.items()
                }
                
                # Sauvegarder chaque segment dans un fichier distinct
                output_file = os.path.join(mode_output_dir, f"{mode}_{os.path.basename(input_file).split('.')[0]}_part_{i+1}.json")
                with open(output_file, 'w') as out_f:
                    json.dump({mode: chunk}, out_f, indent=4)
                
                print(f"Segment {mode} part {i+1} sauvegardé dans {output_file}.")
                
        # Supprimer le fichier original après segmentation
        print(f"Suppression du fichier original : {input_file}.")
        os.remove(input_file)
    else:
        print(f"Le fichier {input_file} fait {file_size_gb:.2f} Go et ne nécessite pas de segmentation.")


def split_large_json_stream(input_file, output_dir, max_size_gb=0, num_parts=5):
    """
    Segmente un fichier JSON volumineux structuré par mode de jeu sans le charger complètement en mémoire.
    Les fichiers JSON sont de la forme :
    {
        'bullet':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'blitz':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'rapid':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]},
        'daily':{'fen':[], 'pgn':[], 'white_rating':[], 'black_rating':[], 'white_result':[], 'black_result':[], 'opening':[]}
    }
    
    Args:
        input_file (str): Chemin vers le fichier JSON à segmenter.
        output_dir (str): Répertoire principal où sauvegarder les fichiers segmentés.
        max_size_gb (float): Taille maximale autorisée pour un fichier JSON avant segmentation (en Go).
        num_parts (int): Nombre de segments à créer pour chaque mode.
        
    Returns:
        None
    """
    file_size_gb = os.path.getsize(input_file) / (1024 ** 3)
    
    if file_size_gb > max_size_gb:
        print(f"Le fichier {input_file} fait {file_size_gb:.2f} Go et sera segmenté en streaming.")
        
        # Création du répertoire principal de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_file, 'rb') as f:
            # Parcourir les clés principales ('bullet', 'blitz', etc.) une par une
            game_modes = ijson.kvitems(f, "")
            for mode, mode_data in game_modes:
                print(f"Traitement du mode : {mode}")
                
                # Créer le dossier spécifique pour ce mode
                mode_output_dir = os.path.join(output_dir, mode)
                os.makedirs(mode_output_dir, exist_ok=True)
                
                # Charger les données de ce mode en segments
                total_entries = len(mode_data['fen'])
                chunk_size = total_entries // num_parts
                
                for i in range(num_parts):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < num_parts - 1 else total_entries
                    
                    # Créer un segment de données
                    chunk = {
                        key: value[start_idx:end_idx]
                        for key, value in mode_data.items()
                    }
                    
                    # Sauvegarder le segment dans un fichier JSON
                    output_file = os.path.join(
                        mode_output_dir,
                        f"{mode}_{os.path.basename(input_file).split('.')[0]}_part_{i+1}.json"
                    )
                    with open(output_file, 'w') as out_f:
                        json.dump({mode: chunk}, out_f, indent=4)
                    
                    print(f"Segment {mode} part {i+1} sauvegardé dans {output_file}.")
        
        # Supprimer le fichier original après traitement
        print(f"Suppression du fichier original : {input_file}.")
        os.remove(input_file)
    else:
        print(f"Le fichier {input_file} fait {file_size_gb:.2f} Go et ne nécessite pas de segmentation.")


def split_json_file_single_mode(input_file, output_dir, num_parts=5):
    """
    Segmente un fichier JSON en plusieurs parties sans le charger entièrement en mémoire.
    
    Args:
        input_file (str): Chemin vers le fichier JSON à segmenter.
        output_dir (str): Répertoire où sauvegarder les fichiers segmentés.
        num_parts (int): Nombre de segments à créer.
        
    Returns:
        None
    """
    # Charger le fichier JSON
    print(f"Chargement du fichier {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Vérification de la structure attendue
    if not isinstance(data, dict) or 'fen' not in data:
        raise ValueError("Le fichier JSON doit contenir une structure de type {'fen':[], ...}")
    
    # Calculer le nombre total d'entrées
    total_entries = len(data['fen'])
    print(f"Nombre total d'entrées : {total_entries}")
    if total_entries == 0:
        raise ValueError("Le fichier JSON ne contient aucune entrée à segmenter.")
    
    # Calculer la taille de chaque segment
    chunk_size = total_entries // num_parts
    print(f"Taille de chaque segment : {chunk_size}")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_parts - 1 else total_entries
        
        # Créer un segment pour ce fichier
        chunk = {key: value[start_idx:end_idx] for key, value in data.items()}
        
        # Sauvegarder le segment dans un fichier JSON
        output_file = os.path.join(
            output_dir,
            f"{os.path.basename(input_file).split('.')[0]}_part_{i+1}.json"
        )
        with open(output_file, 'w') as out_f:
            json.dump(chunk, out_f, indent=4)
        
        print(f"Segment {i+1} sauvegardé dans {output_file}.")



def verify_segmentation(input_file, output_dir, base_filename):
    """
    Vérifie que la segmentation d'un fichier JSON volumineux s'est déroulée correctement.

    Args:
        input_file (str): Chemin vers le fichier JSON d'origine.
        output_dir (str): Répertoire contenant les fichiers segmentés.
        base_filename (str): Nom de base à rechercher dans les fichiers segmentés.

    Returns:
        bool: True si tout est correct, False en cas de problème.
    """
    print("Début de la vérification de la segmentation...")
    # Charger les données originales en mode streaming
    original_counts = {}
    with open(input_file, 'rb') as f:
        game_modes = ijson.kvitems(f, "")
        for mode, mode_data in game_modes:
            original_counts[mode] = len(mode_data['fen'])
            print(f"Mode {mode}: {original_counts[mode]} parties originales.")

    # Vérifier les données segmentées
    segmented_counts = {}
    for mode in original_counts.keys():
        mode_dir = os.path.join(output_dir, mode)
        if not os.path.exists(mode_dir):
            print(f"Erreur : le dossier pour le mode {mode} n'existe pas.")
            return False

        segmented_counts[mode] = 0
        for file in sorted(os.listdir(mode_dir)):
            if file.endswith('.json') and base_filename in file:
                with open(os.path.join(mode_dir, file), 'r') as f:
                    data = json.load(f)
                    segmented_counts[mode] += len(data[mode]['fen'])

        print(f"Mode {mode}: {segmented_counts[mode]} parties segmentées.")

        # Comparer le total
        if original_counts[mode] != segmented_counts[mode]:
            print(f"Erreur : Discrepance pour le mode {mode} (original: {original_counts[mode]}, segmenté: {segmented_counts[mode]}).")
            return False

    print("Vérification terminée : aucune perte ou différence détectée.")
    return True


def verify_segmentation_single_mode(input_file, output_dir, base_filename):
    """
    Vérifie que la segmentation d'un fichier JSON volumineux s'est déroulée correctement
    pour le nouveau format de données.
    
    Args:
        input_file (str): Chemin vers le fichier JSON d'origine.
        output_dir (str): Répertoire contenant les fichiers segmentés.
        base_filename (str): Nom de base à rechercher dans les fichiers segmentés.

    Returns:
        bool: True si tout est correct, False en cas de problème.
    """
    print("Début de la vérification de la segmentation...")

    # Étape 1 : Compter le nombre d'éléments dans chaque clé du fichier original
    original_counts = {}
    with open(input_file, 'r') as f:
        original_data = json.load(f)
        for key in original_data.keys():
            original_counts[key] = len(original_data[key])
            print(f"Clé '{key}': {original_counts[key]} éléments dans le fichier original.")

    # Étape 2 : Vérifier les segments
    segmented_counts = {key: 0 for key in original_counts.keys()}
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.json') and base_filename in file:
            segment_path = os.path.join(output_dir, file)
            with open(segment_path, 'r') as f:
                segment_data = json.load(f)
                for key in segment_data.keys():
                    segmented_counts[key] += len(segment_data[key])
    
    for key in original_counts.keys():
        print(f"Clé '{key}': {segmented_counts[key]} éléments dans les segments.")

    # Étape 3 : Comparer les totaux
    for key in original_counts.keys():
        if original_counts[key] != segmented_counts[key]:
            print(f"Erreur : Discrepance pour la clé '{key}' "
                  f"(original: {original_counts[key]}, segmenté: {segmented_counts[key]}).")
            return False

    print("Vérification terminée : aucune perte ou différence détectée.")
    return True


def create_json_by_elo(input_dir, output_dir, modes=['blitz', 'bullet', 'rapid', 'daily'], elo_bucket_size=100, full_limit=500):
    """
    Crée des JSON organisés par tranche d'Elo sans doublons.

    Args:
        input_dir (str): Répertoire contenant les fichiers segmentés.
        output_dir (str): Répertoire où les nouveaux JSON seront stockés.
        modes (list): Liste des modes à traiter (ex: ['bullet', 'blitz']).
        elo_bucket_size (int): Taille des tranches d'Elo (par défaut 100).
        full_limit (int): Nombre max de parties par tranche pour le JSON 'full'.
    """
    os.makedirs(output_dir, exist_ok=True)

    for mode in modes:
        print(f"Traitement du mode : {mode}")
        mode_dir = os.path.join(input_dir, mode)
        if not os.path.exists(mode_dir):
            print(f"Erreur : le dossier pour le mode {mode} n'existe pas.")
            continue

        # Tranches d'Elo et ensemble pour suivre les doublons
        elo_ranges = defaultdict(int)
        seen_games = set()
        
        # Crée la structure du json final
        full_data = create_game_dict()

        # Parcourir les fichiers du mode
        for file in sorted(os.listdir(mode_dir)):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(mode_dir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                games = data.get(mode, {})

                # Trier les parties dans les tranches d'Elo
                for i, white_rating in enumerate(games.get('white_rating', [])):
                    black_rating = games.get('black_rating', [])[i]
                    
                    # On ne garde que les parties avec des classements serrés
                    if np.abs(white_rating - black_rating) > 300:
                        continue
                    
                    # Prendre la moyenne des Elo des deux joueurs pour classer
                    avg_elo = (white_rating + black_rating) // 2
                    elo_key = (avg_elo // elo_bucket_size) * elo_bucket_size

                    # Générer une clé unique pour la partie
                    game_key = (
                        games.get('pgn', [])[i],
                        games.get('fen', [])[i],
                        white_rating,
                        black_rating
                    )

                    # Vérifier les doublons
                    if game_key in seen_games or elo_ranges[elo_key] >= full_limit:
                        continue

                    # Ajouter la partie aux données finales
                    pgn = games.get('pgn', [])[i]
                    fen = games.get('fen', [])[i]
                    white_result = games.get('white_result', [])[i]
                    black_result = games.get('black_result', [])[i]
                    opening = games.get('opening', [])[i]

                    full_data['fen'].append(fen)
                    full_data['pgn'].append(pgn)
                    full_data['white_rating'].append(white_rating)
                    full_data['black_rating'].append(black_rating)
                    full_data['white_result'].append(white_result)
                    full_data['black_result'].append(black_result)
                    full_data['opening'].append(opening)

                    # Ajouter la clé au set et incrémenter le compteur
                    seen_games.add(game_key)
                    elo_ranges[elo_key] += 1

        output_path = os.path.join(output_dir, f"{mode}_{full_limit}.json")

        # Sauvegarder le JSON
        with open(output_path, 'w') as out_f:
            json.dump(full_data, out_f)
        print(f"full JSON sauvegardé pour {mode} : {output_path}")
        

def get_optimal_workers(worker_ratio=0.75):
    """
    Calcule le nombre optimal de workers pour multiprocessing.Pool.
    
    Args:
        worker_ratio (float): Proportion des cœurs CPU à utiliser (entre 0 et 1).
                              Par défaut, utilise 75% des cœurs disponibles.
    
    Returns:
        int: Nombre optimal de workers.
    """
    try:
        # Obtenir le nombre de cœurs disponibles
        total_cores = multiprocessing.cpu_count()
        
        # Calculer le nombre de workers en fonction du ratio
        workers = max(1, int(total_cores * worker_ratio))
        
        return workers
    except Exception as e:
        print(f"Erreur lors du calcul des workers : {e}")
        return 1  # Valeur par défaut en cas d'erreur


def display_progression_bar(filename, index, total_rows, bar_length=40):
    progression = index / total_rows
    completed_length = int(bar_length * progression)
    bar = '=' * completed_length + '-' * (bar_length - completed_length)
    percentage = round(progression * 100, 2)
    sys.stdout.write(f"\r{filename}: [{bar}] {index}/{total_rows} - {percentage}%")
    sys.stdout.flush()
    

def parse_pgn_with_time(pgn):
    # Expression régulière pour capturer les temps {[%clk X:XX:XX.X]}
    time_pattern = r"\{\[%clk ([0-9:.]+)\]\}"
    # Expression régulière pour capturer les coups en supprimant les temps
    move_pattern = r"([0-9]+\.\s+\S+|\.\.\.\s+\S+)"

    # Extraire les temps
    times = re.findall(time_pattern, pgn)

    # Supprimer les temps du PGN
    clean_pgn = re.sub(time_pattern, '', pgn).strip()

    # Extraire uniquement les coups
    moves = re.findall(move_pattern, clean_pgn)
    moves = [move.split()[-1] for move in moves]  # On garde uniquement le coup, pas le numéro.

    # Construire un DataFrame
    data = {
        "Move Number": [i // 2 + 1 for i in range(len(moves))],  # Numéro du coup
        "Player": ["White" if i % 2 == 0 else "Black" for i in range(len(moves))],  # Joueur
        "Move": moves,
        "Time": times
    }

    df = pd.DataFrame(data)

    return clean_pgn, df


def get_pgn_sequences(pgn):
    """
    Retourne uniquement la partie séquence de coups du pgn
    
    Args:
        pgn (str): str contenant toutes les informations de la partie
    Returns:
        Seulement la séquence de coup "1. c4 {[%clk 63:02:21]} 1... Nf6 {[%clk 7:43:58]} etc..."
    """
    return pgn.split("\n")[-1]


def reconstitute_json(input_dir, output_dir):
    """
    Assemble les JSONS qui ont été segmentés en n part_n et les sauvegarde dans output_dir
    
    Args:
        input_dir (str): chemin qui contient les jsons segmentés
        output_dir (str): chemin vers l'endroit où on veut sauvegarder notre full_json
    Returns:

    """
    full_json = create_evaluated_game_dict()
    
    # Récupère la taille dans le nom du fichier (nombre de parties par tranche de 100 elo)
    size = os.path.basename(input_dir)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde un fichier par mode de jeu
    game_modes = ['blitz', 'bullet', 'daily', 'rapid']
    for mode in game_modes:
        for f in os.listdir(input_dir):
            # On vérifie que c'est un fichier json du mode correspondant
            if f.endswith(".json") and mode in f:
                filename = os.path.join(input_dir, f)
                curr_json = read_json_to_dict(filename)
                
                extend_json(curr_json, full_json)
        
        output_path = os.path.join(output_dir, f"full_evaluated_{mode}_{size}.json")
        save_dict_to_json(output_path, full_json)
   
        
def pd_reconstitute_json(input_dir, output_dir):
    """
    Assemble les JSONS en utilisant pandas qui ont été segmentés en n part_n et les sauvegarde dans output_dir
    
    Args:
        input_dir (str): chemin qui contient les jsons segmentés
        output_dir (str): chemin vers l'endroit où on veut sauvegarder notre full_json
    Returns:

    """
    full_json = pd.DataFrame()
    
    # Récupère la taille dans le nom du fichier (nombre de parties par tranche de 100 elo)
    size = os.path.basename(input_dir)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde un fichier par mode de jeu
    game_modes = ['blitz', 'bullet', 'daily', 'rapid']
    for mode in game_modes:
        for f in os.listdir(input_dir):
            # On vérifie que c'est un fichier json du mode correspondant
            if f.endswith(".json") and mode in f:
                print(f'Traitement de {f} en cours')
                filename = os.path.join(input_dir, f)
                full_json = pd.concat((full_json, pd.read_json(filename)))
        
        output_path = os.path.join(output_dir, f"full_evaluated_{mode}_{size}.json")
        full_json.to_json(output_path, orient='records')


def merge_json_files(input_dir, output_file):
    """
    Merge multiple JSON files from a directory into a single JSON file using streaming to handle memory constraints.

    Args:
        input_dir (str): Directory containing the JSON files to merge.
        output_file (str): Path for the output merged JSON file.
    """
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    
    with open(output_file, 'w') as outfile:
        outfile.write('[')  # Start of a JSON array
        first_file = True
        
        for file_path in json_files:
            with open(file_path, 'rb') as infile:
                parser = ijson.items(infile, 'item')  # Stream JSON items
                for obj in parser:
                    if not first_file:
                        outfile.write(',')  # Add a comma between JSON objects
                    else:
                        first_file = False
                    outfile.write(json.dumps(obj))  # Write the JSON object

        outfile.write(']')  # End of the JSON array

    print(f"Merged {len(json_files)} files into {output_file}.")


def extend_json(curr_json, full_json):
    fen = [curr_json['fen'][k] for k in curr_json['fen']]
    pgn = [curr_json['pgn'][k] for k in curr_json['pgn']]
    white_rating = [curr_json['white_rating'][k] for k in curr_json['white_rating']]
    black_rating = [curr_json['black_rating'][k] for k in curr_json['black_rating']]
    white_result = [curr_json['white_result'][k] for k in curr_json['white_result']]
    black_result = [curr_json['black_result'][k] for k in curr_json['black_result']]
    opening = [curr_json['opening'][k] for k in curr_json['opening']]
    evaluation = [curr_json['evaluation'][k] for k in curr_json['evaluation']]
    best_move = [curr_json['best_move'][k] for k in curr_json['best_move']]
    mate = [curr_json['mate'][k] for k in curr_json['mate']]
    
    # rempli le json
    full_json['fen'].extend(fen)
    full_json['pgn'].extend(pgn)
    full_json['white_rating'].extend(white_rating)
    full_json['black_rating'].extend(black_rating)
    full_json['white_result'].extend(white_result)
    full_json['black_result'].extend(black_result)
    full_json['opening'].extend(opening)
    full_json['evaluation'].extend(evaluation)
    full_json['best_move'].extend(best_move)
    full_json['mate'].extend(mate)

def reconstitute_json_in_gcloud(bucket_name, size):
    """Write and read a blob from GCS using file-like IO"""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your new GCS object
    # blob_name = "storage-object-name"

    full_json = create_evaluated_game_dict()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(storage_client.list_blobs(bucket, prefix=f"evaluated_data/{size}/"))
    
    game_modes = ['blitz', 'bullet', 'daily', 'rapid']
    for mode in game_modes:
        for b in blobs:
            if mode in b.name:
                blob = bucket.blob(b.name)
                with blob.open("r") as f:
                    file = json.load(f)
                print(f'Successfully loaded {b.name}')
                extend_json(file, full_json)
        
        # Upload to GCloud
        upblob = bucket.blob(f"full/full_evaluated_{mode}_{size}")
        upblob.upload_from_string(json.dumps(full_json))
        print(f"file saved to full/full_evaluated_{mode}_{size}")
        

def json_to_parquet(input_dir, output_dir):
    """
    Converti les json en parquet sans conserver la colonne fen
    
    Args:
    input_dir (str): chemin qui contient les jsons segmentés
    output_dir (str): chemin vers l'endroit où on veut sauvegarder notre full_json
    Returns:
    """
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    game_modes = ['blitz', 'bullet', 'daily', 'rapid']
    for mode in game_modes:
        for f in os.listdir(input_dir):
            # On vérifie que c'est un fichier json du mode correspondant
            if f.endswith(".json") and mode in f:
                print(f'Traitement de {f} en cours')
                filename = os.path.join(input_dir, f)
                j_df = pd.read_json(filename)
                j_df = j_df.drop(columns='fen')
                output_path = os.path.join(output_dir, f"{os.path.basename(f).split('.')[0]}.parquet")
                j_df.to_parquet(output_path)
                print(f'file saved to {output_path}')
                

def pd_reconstitute_full_parquet(input_dir, output_dir, mode):
    """
    Assemble les parquets en utilisant pandas qui ont été segmentés en n part_n et les sauvegarde dans output_dir
    
    Args:
        input_dir (str): chemin qui contient les jsons segmentés
        output_dir (str): chemin vers l'endroit où on veut sauvegarder notre full_json
        mode (str): game mode
    Returns:

    """
    full_parquet = pd.DataFrame()
    
    # Récupère la taille dans le nom du fichier (nombre de parties par tranche de 100 elo)
    size = os.path.basename(input_dir.split('/')[-1].split('_')[0])
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde un fichier par mode de jeu
    for f in os.listdir(input_dir):
        # On vérifie que c'est un fichier json du mode correspondant
        if f.endswith(".parquet") and mode in f:
            print(f'Traitement de {f} en cours')
            filename = os.path.join(input_dir, f)
            full_parquet = pd.concat((full_parquet, pd.read_parquet(filename)))
            full_parquet = full_parquet.reset_index(drop=True)
            full_parquet[full_parquet.select_dtypes(np.number).columns] = full_parquet[full_parquet.select_dtypes(np.number).columns].astype('int16')
            
    output_path = os.path.join(output_dir, f"full_evaluated_{mode}_{size}.parquet")
    full_parquet.to_parquet(output_path)


def pd_reconstitue_partial_parquet(input_dir, output_dir, mode):
    """
    Assemble les parquets en utilisant pandas qui ont été segmentés en n part_n et les sauvegarde dans output_dir
    
    Args:
        input_dir (str): chemin qui contient les jsons segmentés
        output_dir (str): chemin vers l'endroit où on veut sauvegarder notre full_json
        mode (str): game mode
    Returns:

    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(input_dir):
        res_df = pd.DataFrame()
        if f.endswith(".parquet") and mode in f:
            print(f'Traitement de {f} en cours')
            filename = os.path.join(input_dir, f)        
            df = pd.read_parquet(filename)
            
            res_df['pgn'] = df["pgn"].apply(lambda x: " ".join(preprocessing.pgn_per_color(x)[0]))
            res_df['white_rating'] = df['white_rating'].astype('int16')
            res_df['black_rating'] = df['black_rating'].astype('int16')
            res_df['time_control'] = df["pgn"].apply(lambda x: preprocessing.func_time_control(x))
            res_df['increment'] = df["pgn"].apply(lambda x: preprocessing.func_increment(x))
            res_df['time_per_move'] = df["pgn"].apply(lambda x: preprocessing.times_per_color(x)[0])

            # Reset index for better memory usage
            del df
            res_df = res_df.reset_index(drop=True)
            
            output_path = os.path.join(output_dir, f"partial_{os.path.basename(f).split('.')[0]}.parquet")
            res_df.to_parquet(output_path)
            print(f'file saved to {output_path}')


def read_parquet_from_gcloud_df(bucket_name, gcloud_path):
    """
    Load a file from gcloud and returns a df
    
    Args:
        bucket_name (str): 
        gcloud_path (str): 

    Returns:
        returns a parquet file as a dataframe 
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcloud_path)
    
    # Read the file into memory
    data = blob.download_as_bytes()
    
    # Load the parquet data into a DataFrame
    df = pd.read_parquet(BytesIO(data))
    
    print(f'df loaded from {gcloud_path}')
    return df
