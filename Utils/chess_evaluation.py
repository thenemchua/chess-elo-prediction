import os
import io

import chess
import chess.engine
import chess.pgn

import pandas as pd

from multiprocessing import Pool
from functools import partial

def analyze_pgn(pgn_string, stockfish_path='./stockfish/stockfish-ubuntu-x86-64-avx2', depth=15):
    """
    Analyse une partie entière en utilisant Stockfish.
    
    Args:
        pgn_string (str): Partie au format PGN sous forme de chaîne.
        stockfish_path (str): Chemin vers l'exécutable Stockfish.
        depth (int): Profondeur d'analyse.
    
    Returns:
        dict: Contient les évaluations et les meilleurs coups pour chaque position.
    """
    # Nettoyer le PGN
    cleaned_pgn = pgn_string.replace("\\n", "\n")
    
    # Charger la partie PGN
    pgn_io = io.StringIO(cleaned_pgn)
    game = chess.pgn.read_game(pgn_io)
    
    if game is None:
        raise ValueError("Impossible de lire le PGN. Assurez-vous que le format est correct.")
    
    board = game.board()
    mates = []
    evaluations = []
    best_moves = []
    
    # Lancer Stockfish et analyser la partie
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for move in game.mainline_moves():
            board.push(move)
            result = engine.analyse(board, chess.engine.Limit(depth=depth))
            
            # Évaluation de la position
            score = result['score'].white()
            if score.is_mate():  # Vérifie s'il y a un mat en n coups
                mate = score.mate()
            else:
                mate = 0
                
            mates.append(mate)
            
            evaluation = score.score()  # Évaluation en centipions
            evaluations.append(evaluation)
            
            # Meilleur coup proposé par Stockfish (s'il existe)
            if 'pv' in result:
                best_move_sequence = [str(m) for m in result['pv']]
                best_moves.append(best_move_sequence)
            else:
                best_moves.append([])  # Aucun meilleur coup trouvé
    
    return {"evaluation": evaluations, "best_move": best_moves, "mate": mates}


def process_with_stockfish(df, stockfish_path='./stockfish/stockfish-ubuntu-x86-64-avx2', depth=15, output_path="evaluated_games.json"):
    """
    Enrichit un DataFrame en ajoutant des colonnes d'évaluation et de meilleur coup Stockfish,
    et sauvegarde le résultat dans un fichier JSON.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les parties (colonne 'pgn').
        stockfish_path (str): Chemin vers l'exécutable Stockfish.
        depth (int): Profondeur d'analyse.
        output_path (str): Chemin de sauvegarde du fichier JSON évalué.
    
    Returns:
        pd.DataFrame: DataFrame enrichi avec les colonnes d'évaluation, de meilleur coup et de mate s'il y en a un.
    """
    evaluations = []
    for index, pgn in enumerate(df['pgn']):
        try:
            eval_result = analyze_pgn(pgn, stockfish_path, depth=depth)
            evaluations.append(eval_result)
        except Exception as e:
            print(f"Erreur lors de l'analyse du PGN à l'index {index}: {e}")
            evaluations.append({"evaluation": None, "best_move": None, "mate": None})
    
    # Ajouter les données évaluées au DataFrame
    df['evaluation'] = [e['evaluation'] for e in evaluations]
    df['best_move'] = [e['best_move'] for e in evaluations]
    df['mate'] = [e['mate'] for e in evaluations]
    
    # Sauvegarder en JSON
    df.to_json(output_path)
    print(f"Fichier évalué sauvegardé dans : {output_path}")
    
    return df


def evaluate_single_file(input_path, output_dir, stockfish_path='./stockfish/stockfish-ubuntu-x86-64-avx2', depth=15):
    """
    Évalue un fichier JSON et sauvegarde le résultat.
    """
    output_path = os.path.join(output_dir, f"evaluated_{os.path.basename(input_path)}")
    
    try:
        print(f"Traitement du fichier : {input_path}")
        
        # Charger le fichier JSON en DataFrame
        df = pd.read_json(input_path)
        
        # Traiter les données avec Stockfish
        process_with_stockfish(df, stockfish_path, depth, output_path)
    except Exception as e:
        print(f"Erreur lors du traitement de {input_path}: {e}")


def evaluate_games_in_directory(input_dir, output_dir, stockfish_path='./stockfish/stockfish-ubuntu-x86-64-avx2', depth=15, workers=4):
    """
    Évalue toutes les parties d'échecs dans un répertoire et sauvegarde les résultats évalués.
    
    Args:
        input_dir (str): Répertoire contenant les fichiers de parties au format JSON.
        output_dir (str): Répertoire pour sauvegarder les fichiers évalués.
        stockfish_path (str): Chemin vers l'exécutable Stockfish.
        depth (int): Profondeur d'analyse.
        workers (int): Nombre de processus à utiliser pour la parallélisation.
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Lister les fichiers JSON et les trier par taille (plus petits en premier)
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".json")
    ]
    files = sorted(files, key=lambda x: os.path.getsize(x))  # Trier par taille
    
    # Parallélisation avec Pool
    with Pool(processes=workers) as pool:
        pool.map(
            partial(evaluate_single_file, output_dir=output_dir, stockfish_path=stockfish_path, depth=depth),
            files
        )