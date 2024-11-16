from collections import defaultdict
import csv
import ijson
import json
import os
import pandas as pd

def create_games_dict():
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

def process_dataframe(df):
    print(f"Traitement d'un batch de taille : {len(df)}")

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

def create_json_by_elo(input_dir, output_dir, modes=['blitz', 'bullet', 'rapid', 'daily'], elo_bucket_size=100, full_limit=50000, medium_limit=10000, small_limit=2000):
    """
    Crée des JSON organisés par tranche d'Elo pour différentes tailles.

    Args:
        input_dir (str): Répertoire contenant les fichiers segmentés.
        output_dir (str): Répertoire où les nouveaux JSON seront stockés.
        modes (list): Liste des modes à traiter (ex: ['bullet', 'blitz']).
        elo_bucket_size (int): Taille des tranches d'Elo (par défaut 100).
        full_limit (int): Nombre max de parties par tranche pour le JSON 'full'.
        medium_limit (int): Nombre max de parties par tranche pour le JSON 'medium'.
        small_limit (int): Nombre max de parties par tranche pour le JSON 'small'.
    """
    os.makedirs(output_dir, exist_ok=True)

    for mode in modes:
        print(f"Traitement du mode : {mode}")
        mode_dir = os.path.join(input_dir, mode)
        if not os.path.exists(mode_dir):
            print(f"Erreur : le dossier pour le mode {mode} n'existe pas.")
            continue

        # Tranches d'Elo
        elo_ranges = defaultdict(list)

        # Parcourir les fichiers du mode
        for file in sorted(os.listdir(mode_dir)):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(mode_dir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                games = data[mode]

                # Trier les parties dans les tranches d'Elo
                for i, white_rating in enumerate(games['white_rating']):
                    black_rating = games['black_rating'][i]
                    pgn = games['pgn'][i]
                    fen = games['fen'][i]
                    white_result = games['white_result'][i]
                    black_result = games['black_result'][i]
                    opening = games['opening'][i]

                    # Prendre la moyenne des Elo des deux joueurs pour classer
                    avg_elo = (white_rating + black_rating) // 2
                    elo_key = (avg_elo // elo_bucket_size) * elo_bucket_size

                    elo_ranges[elo_key].append({
                        'fen': fen,
                        'pgn': pgn,
                        'white_rating': white_rating,
                        'black_rating': black_rating,
                        'white_result': white_result,
                        'black_result': black_result,
                        'opening': opening,
                    })

        # Construire les JSON pour chaque taille
        for size, limit in [('full', full_limit), ('medium', medium_limit), ('small', small_limit)]:
            output_path = os.path.join(output_dir, f"{mode}_{size}.json")
            selected_games = defaultdict(list)

            # Limiter le nombre de parties par tranche
            for elo_key, games in sorted(elo_ranges.items()):
                selected_games[elo_key].extend(games[:limit])

            # Sauvegarder le JSON
            with open(output_path, 'w') as out_f:
                json.dump(selected_games, out_f)
            print(f"{size.capitalize()} JSON sauvegardé pour {mode} : {output_path}")
