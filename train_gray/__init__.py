import path, sys
file_path = path.Path(__file__).abspath()
sys.path.append(file_path.parent.parent.parent)