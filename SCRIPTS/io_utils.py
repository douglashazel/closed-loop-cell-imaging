import msgpack
import pandas as pd

def load_msgpack(path):
    with open(path, 'rb') as f:
        return msgpack.unpack(f, raw=False)

def save_msgpack(data, path):
    with open(path, 'wb') as f:
        msgpack.pack(data, f)

def lum_dict_to_df(data):
    rows = [{'CellID': int(k), **v} for k, v in data.items()]
    df = pd.DataFrame(rows)
    frame_cols = sorted([c for c in df.columns if c.startswith('f')], key=lambda x: int(x[1:]))
    return df[['CellID'] + frame_cols].sort_values('CellID').reset_index(drop=True)

def traj_dict_to_df(data):
    rows = [{'CellID': int(k), **v} for k, v in data.items()]
    df = pd.DataFrame(rows)
    coord_cols = sorted([c for c in df.columns if c != 'CellID'], key=lambda c: (int(c[1:]), c[0]))
    return df[['CellID'] + coord_cols].sort_values('CellID').reset_index(drop=True)

def log_message(log_file_path, message, print_to_console=False):
    with open(log_file_path, 'a') as f:
        f.write(message + '\n')
    if print_to_console:
        print(message)