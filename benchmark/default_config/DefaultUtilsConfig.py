# ----- Storage -----
DEFAULT_LINEAR_STORAGE = {
    
}

DEFAULT_GRAPH_STORAGE = {
    
}

# ----- Display -----
DEFAULT_SCREEN_DISPLAY = {
    'method': 'ScreenDisplay',
    'prefix': '----- Current Memory Start (%s) -----',
    'suffix': '----- Current Memory End -----',
    'key_format': '(%s)',
    'key_value_sep': '\n',
    'item_sep': '\n'
}

DEFAULT_FILE_DISPLAY = {
    'method': 'FileDisplay',
    'prefix': '----- Current Memory Start (%s) -----',
    'suffix': '----- End -----',
    'key_format': '(%s)',
    'key_value_sep': '\n',
    'item_sep': '\n',
    'output_path': 'logs/sample.log'
}

DEFAULT_DISPLAY = DEFAULT_FILE_DISPLAY