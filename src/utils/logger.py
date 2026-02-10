"""Logging utilities for training.

Handles Tensorboard logging, console output, and metrics tracking.
"""

import csv
from pathlib import Path
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")


class Logger:
    """Logger for training metrics."""

    def __init__(self, log_dir, use_tensorboard=True, use_csv=True):
        """Initialize logger.

        Args:
            log_dir (str): Directory for logs
            use_tensorboard (bool): Enable tensorboard logging
            use_csv (bool): Enable CSV logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tensorboard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"Tensorboard logging enabled: {self.log_dir}")
        else:
            self.writer = None

        # CSV logging
        self.use_csv = use_csv
        if self.use_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = self.log_dir / f"metrics_{timestamp}.csv"
            self.csv_file = None
            self.csv_writer = None
            self.csv_fieldnames = None

    def log_episode(self, episode, metrics):
        """Log episode metrics.

        Args:
            episode (int): Episode number
            metrics (dict): Dictionary of metrics to log
        """
        # Console output
        metric_str = " | ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items()])
        print(f"Episode {episode} | {metric_str}")

        # Tensorboard
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"episode/{key}", value, episode)

        # CSV
        if self.use_csv:
            self._write_csv(episode, metrics)

    def log_step(self, step, metrics):
        """Log step-level metrics.

        Args:
            step (int): Step number
            metrics (dict): Dictionary of metrics to log
        """
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"step/{key}", value, step)

    def _write_csv(self, episode, metrics):
        """Write metrics to CSV file.

        Args:
            episode (int): Episode number
            metrics (dict): Metrics dictionary
        """
        # Add episode to metrics
        row = {'episode': episode, **metrics}

        # Initialize CSV on first write
        if self.csv_file is None:
            self.csv_fieldnames = list(row.keys())
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames, restval='')
            self.csv_writer.writeheader()

        # Handle new keys (e.g. eval metrics appearing later)
        new_keys = [k for k in row.keys() if k not in self.csv_fieldnames]
        if new_keys:
            self.csv_fieldnames.extend(new_keys)
            self.csv_file.close()
            existing_rows = []
            with open(self.csv_path, 'r', newline='') as f:
                existing_rows = list(csv.DictReader(f))
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames, restval='')
            self.csv_writer.writeheader()
            self.csv_writer.writerows(existing_rows)

        # Write row
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        """Close logger."""
        if self.use_tensorboard and self.writer:
            self.writer.close()

        if self.use_csv and self.csv_file:
            self.csv_file.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
