import importlib


def get_environment(environment_name):
    return importlib.import_module("chronosrl.environments." + ("go2" if environment_name.startswith("go2") else "jaxgcrl"))
