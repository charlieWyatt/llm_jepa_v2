import logging
import sys


class Logger:
    """Singleton logger instance per GPU rank."""

    _instances: dict[int, 'Logger'] = {}
    _lock = None

    def __new__(cls, rank: int):
        """Ensure only one Logger instance per rank."""
        if rank not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[rank] = instance
        return cls._instances[rank]

    def __init__(self, rank: int):
        """Initialize logger only once per rank."""
        if hasattr(self, '_initialized'):
            return

        self.rank = rank

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )

        logger = logging.getLogger(f"gpu_rank_{rank}")
        logger.setLevel(logging.INFO)
        self.logger = logger

        self._initialized = True

    def log(self, msg: str):
        """Log message from any GPU."""
        self.logger.info(f"[Rank {self.rank}] {msg}")

    def log_only_one_gpu(self, msg: str):
        """Log message only from rank 0."""
        if self.rank == 0:
            self.logger.info(msg)

    @classmethod
    def get_instance(cls, rank: int) -> 'Logger':
        """Get or create logger instance for given rank."""
        return cls(rank)

    @classmethod
    def reset_instances(cls):
        """Reset all instances (useful for testing)."""
        cls._instances.clear()
