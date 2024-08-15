# Gunicorn configuration file
import multiprocessing

port = 4002

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = f"0.0.0.0:{port}"

workers = 1 #(multiprocessing.cpu_count() * 2) + 1
threads = workers

timeout = 120
