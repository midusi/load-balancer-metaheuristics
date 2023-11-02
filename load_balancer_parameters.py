from typing import Optional
from utils import KernelName, OptimizerName, ClusteringAlgorithm, ClusteringScoringMethod


class LoadBalancerParameters:
    """Super class for Model's parameters to predict execution times."""
    number_of_samples: Optional[int]

    def __init__(self, number_of_samples: Optional[int]):
        self.number_of_samples = number_of_samples


class SVMParameters(LoadBalancerParameters):
    kernel: KernelName
    optimizer: OptimizerName

    def __init__(self, kernel: KernelName, optimizer: OptimizerName, number_of_samples: Optional[int] = None):
        super(SVMParameters, self).__init__(number_of_samples)

        self.kernel = kernel
        self.optimizer = optimizer

    def __str__(self):
        return (f'SVMParameters(number_of_samples={self.number_of_samples}, kernel="{self.kernel}", '
                f'optimizer="{self.optimizer}")')


class ClusteringParameters(LoadBalancerParameters):
    algorithm: ClusteringAlgorithm
    number_of_clusters: int
    scoring_method: ClusteringScoringMethod

    def __init__(self, algorithm: ClusteringAlgorithm, number_of_clusters: int, scoring_method: ClusteringScoringMethod,
                 number_of_samples: Optional[int] = None):
        super(ClusteringParameters, self).__init__(number_of_samples)

        self.algorithm = algorithm
        self.number_of_clusters = number_of_clusters
        self.scoring_method = scoring_method

    def __str__(self):
        return (f'ClusteringParameters(number_of_samples={self.number_of_samples}, algorithm="{self.algorithm}", '
                f'number_of_clusters="{self.number_of_clusters}, scoring_method="{self.scoring_method}")')


class RFParameters(LoadBalancerParameters):
    number_of_trees: int

    def __init__(self, number_of_trees: int, number_of_samples: Optional[int] = None):
        super(RFParameters, self).__init__(number_of_samples)

        self.number_of_trees = number_of_trees

    def __str__(self):
        return f'ClusteringParameters(number_of_samples={self.number_of_samples}, trees={self.number_of_trees})'
