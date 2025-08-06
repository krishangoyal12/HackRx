import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1  # Keep at 1 for memory constraints
threads = 2  # Increased to 2 threads to handle concurrent requests
timeout = int(os.environ.get('RENDER_TIMEOUT_SECONDS', 300))  # Use environment variable
preload_app = True  # CHANGED to True to load models at startup
max_requests = 10  # Lowered to restart workers more frequently
max_requests_jitter = 3
worker_class = "sync"  # Use sync workers for ML workloads