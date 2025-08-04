import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1  # Keep at 1 for memory constraints
threads = 1  # Reduce threads to save memory
timeout = 300  # Increase timeout to 5 minutes
preload_app = False  # Keep this to save memory
max_requests = 1000  # Restart workers periodically
max_requests_jitter = 100
worker_class = "sync"  # Use sync workers for ML workloads