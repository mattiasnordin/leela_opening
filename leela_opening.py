import pandas as pd
import re
import io
import numpy as np
import json
import chess
import chess.pgn
import chess.syzygy
import os
import shutil
import time
import datetime
from urllib.request import Request, urlopen
import csv
from bs4 import BeautifulSoup
from urllib.error import HTTPError
import gzip
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from operator import add
from tqdm import tqdm

file = 'c:\\users\\menor\\downloads\\9999.pgn'



def pgn_max_36_plies(df, pgn):
    """
    Longest opening in eco.json is 36 plies (18 moves)
    Function adds a column, 'opening_pgn', with at most the first 36 plies
    """
    df2 = df[pgn].str.extract('(1\..*?) 19\.')
    df2.rename(columns={0: 'opening_pgn'}, inplace=True)
    df_merge = pd.merge(df, df2, left_index=True, right_index=True)
    df_merge['opening_pgn'] = np.where(pd.isnull(
    df_merge['opening_pgn']), df_merge[pgn], df_merge['opening_pgn'])

    return df_merge


def classify_fen(fen, ecodb):
    """
    Searches a JSON file with Encyclopedia of Chess Openings (ECO) data to
    check if the given FEN matches an existing opening record

    Returns a classification

    A classfication is a dictionary containing the following elements:
        "code":         The ECO code of the matched opening
        "desc":         The long description of the matched opening
        "path":         The main variation of the opening
    """
    classification = {}
    classification["code"] = ""
    classification["desc"] = ""
    classification["path"] = ""

    for opening in ecodb:
        if opening['f'] == fen:
            classification["code"] = opening['c']
            classification["desc"] = opening['n']
            classification["path"] = opening['m']

    return classification


def eco_fen(board):
    """
    Takes a board position and returns a FEN string formatted
    for matching with eco.json
    """
    board_fen = board.board_fen()
    castling_fen = board.castling_xfen()

    if board.turn:  # If white to move
        to_move = 'w'
    else:
        to_move = 'b'

    fen = board_fen + " " + to_move + " " + castling_fen
    return fen


def classify_opening(game):
    """
    Takes a game and adds an ECO code classification for the opening
    Returns the classified game and root_node, which is the node where
    the classification was made
    """

    eco_code = ''
    opening_name = ''
    ecofile = os.path.join(ecofile_path)
    ecodata = json.load(open(ecofile, 'r'))

    node = game.end()

    while not node == game.root():
        node_parent = node.parent
        try:
            board = board_parent
        except NameError:
            board = node.board()
        board_parent = node_parent.board()
        board_turn = board.turn
        fen = eco_fen(board)
        classification = classify_fen(fen, ecodata)

        if classification["code"] != "" and eco_code == '':
            # Add some comments classifying the opening
            eco_code = classification["code"]
            opening_name = classification["desc"]

        node = node_parent

    return {'ECO code': eco_code, 'Opening name': opening_name}


def number_pieces(fen):
    str_split = fen.split(' ', 1)
    match = re.findall('r|n|b|q|k|p', str_split[0], flags = re.IGNORECASE)
    return len(match)
    

def update_nr_pieces(board, move, nr_pieces):
    if board.is_capture(move):
        nr_pieces = nr_pieces + 1
        return nr_pieces
    else:
        return nr_pieces


def syzygy_lookup(board, white_turn, tablebases):
    if white_turn:
        return tablebases.probe_wdl(board)
    else:
        return -1 * tablebases.probe_wdl(board)


def tablebase_result(board, white_turn, tablebases):
    t1 = time.time()
    syz = syzygy_lookup(board, white_turn, tablebases)
    syzygy_time = time.time() - t1
    if syz == 2:
        tablebase_eval = 1
    elif syz == -2:
        tablebase_eval = -1
    elif syz >= -1 and syz <= 1:
        tablebase_eval = 0
    else:
        raise ValueError

    return tablebase_eval, syzygy_time


def parse_pgn_syzygy(game):
    t1 = 0
    node = game.end()
    board = node.board()

    result_dict = {'white_checkmate': 0,
                   'black_checkmate': 0,
                   'stalemate': 0,
                   'insufficient_material': 0,
                   'threefold_repetition': 0,
                   'fifty_moves': 0}

    if board.is_checkmate() and not board.turn:
        result_dict['white_checkmate'] = 1

    if board.is_checkmate() and board.turn:
        result_dict['black_checkmate'] = 1

    if board.is_stalemate():
        result_dict['stalemate'] = 1

    if board.is_insufficient_material():
        result_dict['insufficient_material'] = 1

    if board.can_claim_threefold_repetition():
        result_dict['threefold_repetition'] = 1

    if board.can_claim_fifty_moves():
        result_dict['fifty_moves'] = 1

    if board.turn:
        result_dict['plies'] = (board.fullmove_number - 1) * 2

    if not board.turn:
        result_dict['plies'] = board.fullmove_number * 2 - 1
        
    t1 = time.time() - t1
#    fen_end = board.board_fen()
#    nr_pieces = number_pieces(fen_end)
#
#    tablebase_dict = {'white_winning_to_drawing': 0,
#                      'white_winning_to_loosing': 0,
#                      'white_drawing_to_loosing': 0,
#                      'black_winning_to_drawing': 0,
#                      'black_winning_to_loosing': 0,
#                      'black_drawing_to_loosing': 0,
#                      'nr_tablebase_positions': 0}
#
#    while nr_pieces <= 6:
#        a1 = time.time()
#        board = node.board()
#        node_parent = node.parent
#        board_parent = node_parent.board()
#        move = node.move
#        board_turn = board.turn
#        t1 += time.time() - a1
#
#        nr_pieces = update_nr_pieces(board_parent, move, nr_pieces)
#
#        try:
#            tablebase_eval_parent = tablebase_eval
#        except UnboundLocalError:
#            pass
#
#
#        tablebase_eval, t2 = tablebase_result(board, board_turn, tablebases)
#        t1 += t2
#
#        try:
#            if tablebase_eval == 1 and tablebase_eval_parent == 0:
#                tablebase_dict['white_winning_to_drawing'] += 1
#
#            if tablebase_eval == 1 and tablebase_eval_parent == -1:
#                tablebase_dict['white_winning_to_loosing'] += 1
#
#            if tablebase_eval == 0 and tablebase_eval_parent == -1:
#                tablebase_dict['white_drawing_to_loosing'] += 1
#
#            if tablebase_eval == -1 and tablebase_eval_parent == 0:
#                tablebase_dict['black_winning_to_drawing'] += 1
#
#            if tablebase_eval == -1 and tablebase_eval_parent == 1:
#                tablebase_dict['black_winning_to_loosing'] += 1
#
#            if tablebase_eval == 0 and tablebase_eval_parent == 1:
#                tablebase_dict['black_drawing_to_loosing'] += 1
#
#        except UnboundLocalError:
#            pass
#
#        tablebase_dict['nr_tablebase_positions'] += 1
#
#        node = node_parent

#    endgame_dict = {**result_dict, **tablebase_dict}

    return result_dict, t1


def eco_codes(df, eco):
    df['eco_letter'] = df[eco].str.slice(0, 1)
    df['eco_number'] = df[eco].str.slice(1, )
    df['eco_number'] = pd.to_numeric(df['eco_number'])

    def f(x):
        if (x['eco_letter'] == 'B' and x['eco_number'] >= 10 and
            x['eco_number'] <= 19): return "caro_kann"
        elif (x['eco_letter'] == 'B' and x['eco_number'] >= 20 and
              x['eco_number'] <= 99): return "sicilian"
        elif (x['eco_letter'] == 'C' and x['eco_number'] >= 0 and
              x['eco_number'] <= 19): return "french"
        elif (x['eco_letter'] == 'C' and x['eco_number'] >= 20 and
              x['eco_number'] <= 99): return "double_king_pawn"
        elif (x['eco_letter'] == 'D' and x['eco_number'] >= 0 and
              x['eco_number'] <= 69): return "double_queen_pawn"
        elif (x['eco_letter'] == 'D' and x['eco_number'] >= 70 and
              x['eco_number'] <= 99): return "grünfeld"
        elif (x['eco_letter'] == 'E' and x['eco_number'] >= 0 and
              x['eco_number'] <= 99): return "indian"
        elif (x['eco_letter'] == 'A' and x['eco_number'] >= 80 and
              x['eco_number'] <= 99): return "dutch"
        elif (x['eco_letter'] == 'A' and x['eco_number'] >= 10 and
              x['eco_number'] <= 39): return "english"
        elif (x['eco_letter'] == 'A' and
             ((x['eco_number'] >= 43 and x['eco_number'] <= 44) or
             (x['eco_number'] >= 56 and x['eco_number'] <= 79))):
            return "benoni"
        elif (x['eco_letter'] == 'B' and x['eco_number'] >= 0 and
              x['eco_number'] <= 9): return "other_e4"
        elif (x['eco_letter'] == 'A' and x['eco_number'] >= 0 and
              x['eco_number'] <= 9): return "non_e4_d4_c4"
        elif (x['eco_letter'] == 'A' and
             ((x['eco_number'] >= 40 and x['eco_number'] <= 42) or
             (x['eco_number'] >= 45 and x['eco_number'] <= 55))):
            return "other_openings"
        else:
            return "unknown opening"

    df['opening_name'] = df.apply(f, axis=1)

    return df


def generate_opening_endgame_df(df):
    df['nr_tablebase_positions'] = np.where(df['nr_tablebase_positions'] == 0,
                                   0, df['nr_tablebase_positions'] - 1)
    # Counts if position after the last move is a tablebase position. Should be
    # counted before the last move. Correcting here.
    df['white_blunder'] = (df['white_drawing_to_loosing'] +
                           df['white_winning_to_drawing'] +
                           df['white_winning_to_loosing'])

    df['black_blunder'] = (df['black_drawing_to_loosing'] +
                           df['black_winning_to_drawing'] +
                           df['black_winning_to_loosing'])

    df['blunder'] = df['white_blunder'] + df['black_blunder']

    return df


def promotions(df, piece_tuples):
    for piece_tuple in piece_tuples:
        piece_name = piece_tuple[0]
        piece_short = piece_tuple[1]
        df[piece_name + '_promotions'] = df['pgn'].str.count(
            '\=' + piece_short)
        df['white_' + piece_name + '_promotions'] = df['pgn'].str.count(
            '\.\S*\=' + piece_short)
        df['black_' + piece_name + '_promotions'] = (
            df[piece_name + '_promotions'] -
            df['white_' + piece_name + '_promotions'])
    df['white_promotions'] = (df['white_queen_promotions'] +
                              df['white_rook_promotions'] +
                              df['white_bishop_promotions'] +
                              df['white_knight_promotions'])

    df['black_promotions'] = (df['black_queen_promotions'] +
                              df['black_rook_promotions'] +
                              df['black_bishop_promotions'] +
                              df['black_knight_promotions'])

    df['white_under_promotions'] = (df['white_promotions']
                                    - df['white_queen_promotions'])

    df['black_under_promotions'] = (df['black_promotions']
                                    - df['black_queen_promotions'])
    
    return df


def analyze_pgn_file(file):
    with open(file, 'r') as pgn_file:
        pgn_cont = pgn_file.read()
    
    re_pgn = re.compile('(\[.*?\])\n\n(1\..*?)\n\n', flags=re.DOTALL)
    re_white = re.compile('\[White "lc0.net.([0-9]*?)"\]')
    re_black = re.compile('\[Black "lc0.net.([0-9]*?)"\]')
    re_result = re.compile('\[Result "(.*?)"\]')
    
    pgn_list = re.findall(re_pgn, pgn_cont)
    
    out = []
    for pgn in pgn_list:
        white = re.findall(re_white, pgn[0])[0]
        black = re.findall(re_black, pgn[0])[0]
        result = re.findall(re_result, pgn[0])[0]
        out.append({'white': white,
                    'black': black,
                    'result': result,
                    'pgn': pgn[1]})
        
    df = pd.DataFrame(out)
    return df


def extract_pgn_data(df, pgn, out_file):
    df['white_kingside_castle'] = df[pgn].str.match('.*?\.O-O .*?')
    df['black_kingside_castle'] = df[pgn].str.match('.*? O-O .*?')

    df['white_queenside_castle'] = df[pgn].str.match('.*?\.O-O-O.*?')
    df['black_queenside_castle'] = df[pgn].str.match('.*? O-O-O.*?')

    df['white_no_castle'] = 1 - (
        df['white_kingside_castle'] + df['white_queenside_castle'])
    df['black_no_castle'] = 1 - (
        df['black_kingside_castle'] + df['black_queenside_castle'])

    df['white_first_move'] = df[pgn].str.extract('1\.(\S*) ')

    piece_tuples = [('queen', 'Q'), ('rook', 'R'),
                    ('knight', 'N'), ('bishop', 'B')]

    df = promotions(df, piece_tuples)
    df = df.drop('pgn', axis=1)
    
    df.to_csv(out_file, mode='a', header=False, index=False)

    return df


def MatchesMetadata():
    link = 'http://training.lczero.org/matches/'
    req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    source = urlopen(req).read()
    soup = BeautifulSoup(source, 'html.parser')
    table = soup.find('table', attrs={'class':'table table-striped table-sm'})
    table_head = table.find('thead')
    headers = table_head.find_all('th')
    headers = [ele.text.strip() for ele in headers]
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    data = []

    for row in reversed(rows):
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        i = 0
        match_dict = {}

        for col in cols:
            match_dict[headers[i]] = col
            i += 1

        data.append(match_dict)
        
    match_df = pd.DataFrame(data)
    match_df.rename(columns={'Id': 'match_id'}, inplace=True)
    match_df.rename(columns={'Candidate': 'Candidate ID'}, inplace=True)
    match_df.rename(columns={'Current': 'Current ID'}, inplace=True)
    match_df['match_id'] = pd.to_numeric(match_df['match_id'])
    match_df['Candidate ID'] = pd.to_numeric(match_df['Candidate ID'])

    return match_df


def MatchIdNotAnalyzed(df, file):
    df_analyzed = pd.read_csv(file)
    df_analyzed = df_analyzed[['match_id']]
    df_m = pd.merge(df, df_analyzed, on='match_id', how='left',
                    validate='1:1', indicator=True)
    df_not_analyzed = df_m[df_m['_merge'] == 'left_only']
    return df_not_analyzed


def DownloadPgn(match_id):
    url1 = 'http://data.lczero.org/files/match_pgns/'
    url = url1 + '1/' + str(match_id) + '.pgn'
    filename = os.path.join(pgn_folder, str(match_id) + '.pgn')
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req) as response, open(filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    except HTTPError:
        try:
            url = url1 + '2/' + str(match_id) + '.pgn'
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req) as response, open(filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except HTTPError:
            pass
    
    return filename


def validated_merge(df1, df2, merge_key, validation_type, how_param):
    df = pd.merge(df1, df2, on=merge_key, validate=validation_type,
                  indicator=True, how=how_param)
    vals = df['_merge'].value_counts()
    assert (vals['left_only'] == 0 and
            vals['right_only'] == 0), 'merge failed!'
    df = df.drop('_merge', axis=1)
    return df


def CompilePgnData(df_inp, pgn, analyzed_file):
    out = []
    df = pgn_max_36_plies(df_inp, pgn)
    session_games = df.shape[0]

    for i in tqdm(range(session_games)):
        opening_pgn = io.StringIO(
                df.iloc[i, df.columns.get_loc('opening_pgn')])
        pgn_io = io.StringIO(df.iloc[i, df.columns.get_loc(pgn)])
        game_opening = chess.pgn.read_game(opening_pgn)
        game = chess.pgn.read_game(pgn_io)
        opening_dict = classify_opening(game_opening)
        
        endgame_dict, syzygy_time = parse_pgn_syzygy(game)
        obs_dict = {**opening_dict, **endgame_dict}

        out.append(obs_dict)

    df2 = pd.DataFrame(out)
    # df2 = generate_opening_endgame_df(df2)
    df2 = eco_codes(df2, 'ECO code')
    df2.to_csv(analyzed_file, mode='a', header=False, index=False)
    
    return df2


def pivot_wide(df_inp, varlist, varname):

    out_dict = {}
    N = 0
    for var in varlist:
    
        nr = df_inp[df_inp[varname] == var].shape[0]
        out_dict[var] = [nr]
        N += nr
        
    df = pd.DataFrame(out_dict)
    
    for var in varlist:
        df[var] = df[var] / N

    return df


def agg_other_df(df):
    df_agg = df[['white_kingside_castle', 'black_kingside_castle',
                 'white_queenside_castle', 'black_queenside_castle',
                 'white_first_move', 'queen_promotions',
                 'white_queen_promotions', 'black_queen_promotions',
                 'rook_promotions', 'white_rook_promotions',
                 'black_rook_promotions', 'knight_promotions',
                 'white_knight_promotions', 'black_knight_promotions',
                 'bishop_promotions', 'white_bishop_promotions',
                 'black_bishop_promotions', 'white_promotions',
                 'black_promotions', 'white_under_promotions',
                 'black_under_promotions', 'black_checkmate',
                 'fifty_moves', 'insufficient_material',
                 'plies', 'stalemate', 'threefold_repetition',
                 'white_checkmate', 'white_no_castle', 'black_no_castle']]

    df_agg = df_agg.agg(['mean'])
    
    df_agg.reset_index(drop=True, inplace=True)

    return df_agg


def backup_files(orig_folder, backup_folder):
    ''' Back up files in output folder '''
    for file in os.listdir(orig_folder):
        shutil.copy2(os.path.join(orig_folder, file), backup_folder)


def GenerateAggregateData(df):

    opening_list = ['caro_kann', 'sicilian', 'french',
                    'double_king_pawn', 'double_queen_pawn',
                    'grünfeld', 'indian', 'dutch', 'english',
                    'benoni', 'other_e4', 'non_e4_d4_c4', 'other_openings']
    
    df_eco_wide = pivot_wide(df, opening_list, 'opening_name')

    opening_move = ['a3', 'a4', 'b3', 'b4', 'c3', 'c4', 'd3', 'd4', 'e3', 'e4',
                    'f3', 'f4', 'g3', 'g4', 'h3', 'h4', 'Na3', 'Nc3', 'Nf3',
                    'Nh3']

    df_agg_other_first_move = pivot_wide(df, opening_move, 'white_first_move')
    df_agg_other = agg_other_df(df)

    df_agg = df_eco_wide.join(df_agg_other_first_move)
    df_agg = df_agg.join(df_agg_other)

    df_agg['draw'] = (df_agg['threefold_repetition'] + df_agg['stalemate'] +
    	              df_agg['fifty_moves'] + df_agg['insufficient_material'])

    df_agg['sum_n'] = df.shape[0]
    
    return df_agg


def AnalyzeMatch(match_id):
    file = os.path.join(pgn_folder, str(match_id) + '.pgn')
    
    df_analyze_pgn = analyze_pgn_file(file)
    
    df_pgn = df_analyze_pgn[['pgn']]
    
    df_opening_endgame = CompilePgnData(df_pgn, 'pgn', opening_file)
    df_castle_first_move = extract_pgn_data(df_pgn, 'pgn', extract_pgn_file)
    
    df_out = df_opening_endgame.join(df_castle_first_move)
    
    df_agg = GenerateAggregateData(df_out)
    df_agg['match_id'] = match_id
    df_agg.to_csv(match_agg_file, mode='a', header=False, index=False)
    
    os.remove(file)
    
    return df_out.shape[0]


def update_google_spreadsheet(df, sheet):
    df_list = [df.columns]
    for row in df.iterrows():
        index, data = row
        df_list.append(data.tolist())

    sheet.clear()
    cell_list = sheet.range(1, 1, df.shape[0]+1, df.shape[1])
    flat_list = [item for sublist in df_list for item in sublist]

    i = 0
    for cell in cell_list:
        if str(flat_list[i]) == 'nan':
            cell.value = ""
        else:
            cell.value = flat_list[i]
        i += 1

    sheet.update_cells(cell_list, value_input_option='RAW')


project_folder = 'D:\\Leela'
output_folder = os.path.join(project_folder, 'testnet\\output\\')
ecofile_path = os.path.join(project_folder, 'code\\eco\\eco.json')
backup_folder = os.path.join(project_folder, 'testnet\\backup\\')
match_agg_file = os.path.join(output_folder, 'full_agg_data60.csv')
pgn_folder = os.path.join(project_folder, 'testnet\\pgn\\')
opening_file = os.path.join(output_folder, 'opening_data60.csv')
# syzygy_folder = os.path.join(project_folder, 'Syzygy_6piece\\wdl')
extract_pgn_file = os.path.join(output_folder, 'pgn_extract60.csv')
json_file = os.path.join(project_folder, 'code\\client_secret.json')

backup_files(output_folder, backup_folder)

match_df = MatchesMetadata()
match_df = match_df[(match_df['Pass']=='true') | (match_df['Pass']=='test')]
match_df60 = match_df[(match_df['Candidate ID'] > 60000) &
                      (match_df['Candidate ID'] < 70000)]
df_not_analyzed = MatchIdNotAnalyzed(match_df60, match_agg_file)

analyze_list = df_not_analyzed['match_id'].tolist()
# empty_list = [1600, 5718, 5720, 7325, 7419, 8366, 9470, 9471]
# analyze_list = list(set(analyze_list) - set(empty_list))

# tablebases_glob = chess.syzygy.open_tablebases(syzygy_folder)

analyzed = 0
for nr in tqdm(range(len(analyze_list))):
    match_id = analyze_list[nr]
    t1 = time.time()
    try:
        DownloadPgn(match_id)
        games_analyzed = AnalyzeMatch(match_id)
        delta_time = time.time() - t1
        analysis_time = str(datetime.timedelta(seconds=delta_time))
        games_per_sec = games_analyzed / delta_time
    #except KeyError:
    #    print('File does not exist at all', match_id)
    except FileNotFoundError:
        print('File does not exist yet', match_id)
    analyzed += 1
    

full_df = pd.read_csv(match_agg_file)
df_merge = pd.merge(full_df, match_df60, on='match_id', how='inner',
                    validate='1:1')

df_merge.rename(columns={'Candidate ID': 'network_id'}, inplace=True)
df_merge = df_merge.groupby('network_id').agg(['mean'])
df_merge.columns = df_merge.columns.get_level_values(0)
df_merge = df_merge.reset_index()

df_merge.sort_values(by='network_id', inplace=True)

df_agg60 = df_merge[(df_merge['network_id'] >= 60000) &
           (df_merge['network_id'] <= 64999)]


scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name(json_file, scope)
client = gspread.authorize(creds)

worksheet60 = client.open_by_url("https://docs.google.com/spreadsheets/d/12kSavLSwyYXNLJFycjOQmcjWnW_YoKIxuhtIV-acQlA/edit#gid=0").sheet1

update_google_spreadsheet(df_agg60, worksheet60)
