import random


def cxGauss(ind1, ind2, mu, sigma, thre):
    """
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param thre: threthold for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.gauss(mu, sigma) < thre:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2
