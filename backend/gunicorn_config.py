import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1  # Minimum workers to save memory
threads = 2
timeout = 120  # Longer timeout for ML operations
preload_app = False  # Important: Don't preload to save memory