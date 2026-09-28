from chronosrl.algorithms import chronosrl, srl, crl, accrl


ALGORITHMS = {
    "chronosrl": (chronosrl.get_config, chronosrl.ChronoSRL),
    "srl": (srl.get_config, srl.SRL),
    "crl": (crl.get_config, crl.CRL),
    "accrl": (accrl.get_config, accrl.ACCRL),
}
