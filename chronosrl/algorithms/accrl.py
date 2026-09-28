from chronosrl.algorithms.crl import CRL, get_config as get_crl_config


def get_config(algorithm_name):
    config = get_crl_config(algorithm_name)

    config.action_chunk_length = 3
    config.nr_sgd_batches = -1

    return config


ACCRL = CRL
