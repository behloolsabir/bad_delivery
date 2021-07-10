import toml

def getConfig():
    filepath = '../config.toml'
    with open(filepath, "r") as f:
        return toml.load(f)
