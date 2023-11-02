import random
from typing import Union, List, Tuple
import numpy as np
import logging

# Habilito los loggers
logging.getLogger().setLevel(logging.INFO)

RANDOM_MIN_MAX = 32.768


def ackley(x_and_y):
    x, y = x_and_y
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) \
           - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


def compute_cross_validation(x_and_y, test_function) -> float:
    """
    Computa una validacion cruzada calculando el accuracy
    :param: x_and_y:
    :return: Promedio del accuracy obtenido en cada fold del CrossValidation
    """
    return test_function(x_and_y)


def get_best(
        subsets: np.ndarray,
        fitness_values: Union[np.ndarray, List[float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtiene el mayor valor de un conjunto de fitness"""
    best_idx = np.argmin(fitness_values)  # Mantengo el idx para evitar comparaciones ambiguas
    return best_idx, subsets[best_idx], fitness_values[best_idx]


def get_random_x_and_y():
    return random.uniform(-RANDOM_MIN_MAX, RANDOM_MIN_MAX), random.uniform(-RANDOM_MIN_MAX, RANDOM_MIN_MAX)


def binary_black_hole_test(
        n_stars: int,
        n_iterations: int,
        test_function,
        debug: bool = False
) -> Tuple[float, Tuple[float, float], Tuple[float, float], float]:
    """
    Computa la metaheuristica Binary Black Hole sacada del paper
    TODO: si funciona esto hay que volver a demostrar con una solucion vectorizada
    :param test_function:
    :param n_stars: Number of stars
    :param n_iterations: Number of iterations
    :param debug: If True logs everything is happening inside BBHA
    :return: Initial fitness value, a tuple with initial X and Y, a tuple with final X, and Y and the final fitness value
    """
    # Preparo las estructuras de datos
    stars_subsets = np.empty((n_stars, 2))
    stars_fitness_values = np.empty((n_stars,), dtype=float)

    # Inicializo las estrellas con sus subconjuntos y sus valores fitness
    if debug:
        logging.info('Initializing stars...')
    for i in range(n_stars):
        x, y = get_random_x_and_y()
        stars_subsets[i] = (x, y)  # Inicializa 'Population'
        stars_fitness_values[i] = compute_cross_validation(stars_subsets[i], test_function)

    # El que mejor fitness tiene es el Black Hole
    black_hole_idx, black_hole_subset, black_hole_fitness = get_best(stars_subsets, stars_fitness_values)

    # Gets initial values
    initial_value = black_hole_fitness
    (initial_x, initial_y) = black_hole_subset
    (best_x, best_y) = black_hole_subset

    if debug:
        logging.info(f'Black hole starting as star at index {black_hole_idx} with fitness {black_hole_fitness}')

    # Iteraciones
    for i in range(n_iterations):
        if debug:
            logging.info(f'Iteration {i + 1}/{n_iterations}')
        for a in range(n_stars):
            # Si es la misma estrella que se convirtio en agujero negro, no hago nada
            if a == black_hole_idx:
                continue

            # Compute the current star fitness
            current_star_subset = stars_subsets[a]
            current_fitness = compute_cross_validation(current_star_subset, test_function)

            # Si la estrella tiene mejor fitness que el agujero negro, hacemos swap
            if current_fitness < black_hole_fitness:
                if debug:
                    logging.info(f'Changing Black hole for star {a},'
                                 f' BH fitness -> {black_hole_fitness} | Star {a} fitness -> {current_fitness}')
                black_hole_idx = a
                black_hole_subset, current_star_subset = current_star_subset, black_hole_subset
                black_hole_fitness, current_fitness = current_fitness, black_hole_fitness
                (best_x, best_y) = black_hole_subset

            # Calculo el horizonte de eventos
            event_horizon = black_hole_fitness / np.sum(stars_fitness_values)

            # Me fijo si la estrella cae en el horizonte de eventos
            dist_to_black_hole = np.linalg.norm(black_hole_subset - current_star_subset)  # Dist. Euclidea
            if dist_to_black_hole < event_horizon:
                stars_subsets[a] = get_random_x_and_y()

        # Actualizo de manera binaria los subsets de cada estrella
        for a in range(n_stars):
            # Salteo el agujero negro
            if black_hole_idx == a:
                continue

            x_old = stars_subsets[a][0]
            x_new = x_old * random.uniform(0, 1) * (stars_subsets[black_hole_idx][0] - x_old)

            y_old = stars_subsets[a][1]
            y_new = y_old * random.uniform(0, 1) * (stars_subsets[black_hole_idx][1] - y_old)
            stars_subsets[a] = (x_new, y_new)

    return initial_value, (initial_x, initial_y), (best_x, best_y), black_hole_fitness


def main():
    results = []
    for i in range(20):
        initial_value, (initial_x, initial_y), (best_x, best_y), black_hole_fitness = binary_black_hole_test(
            n_stars=1000,
            n_iterations=1000,
            test_function=ackley,
            debug=False
        )
        results.append(black_hole_fitness)

        # Should be equal to 0.0
        logging.info(f'Iteration {i + 1} Ackley final fitness -> {black_hole_fitness} in [{round(best_x, 2)}, '
                     f'{round(best_y, 2)}] \n'
                     f'\tInitial was {initial_value} in [{round(initial_x, 2), round(initial_y, 2)}]')

    mean_res = np.mean(results)
    assert mean_res == 0.0


if __name__ == '__main__':
    main()
