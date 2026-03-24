import http.server
import socketserver
import threading
import time
import sys
import random
from tqdm import tqdm

# --- Configuration ---
PORT = 8080
FAKE_DATASETS = [
    ("training_set_v4_high_res.zip", 450),
    ("imagenet_full_backup.tar.gz", 820),
    ("global_weather_historical_data.csv", 310),
    ("large_scale_text_corpus_2024.txt", 1200),
    ("synthetic_images_v2.zip", 600),
    ("audio_samples_collection_2024.tar.gz", 750),
    ("video_dataset_hd_2025.zip", 1424),
    ("multimodal_data_2025.tar.gz", 2300),
]

class SilentServer(http.server.SimpleHTTPRequestHandler):
    """A server that doesn't log requests to the terminal to stay hidden."""
    def log_message(self, format, *args):
        return

def run_server():
    """Starts the background HTTP server."""
    with socketserver.TCPServer(("", PORT), SilentServer) as httpd:
        print(f"# Initializing background services...") # Vague enough to be safe
        httpd.serve_forever()

def start_prank():
    # 1. Start the 'malicious' background server
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print(f"Authenticating with Kaggle API... Success.")
    # print the IP address of the server to make it look more real
    ip_add = http.server.socket.gethostbyname(http.server.socket.gethostname())
    print(f"Connected to Kaggle API from IP: {ip_add}")
    print(f"Destination: /usr/local/datasets/cache/\n")

    try:
        # 2. Loop through fake downloads
        for name, size in FAKE_DATASETS:
            print(f"Downloading {name}")
            # tqdm creates the realistic progress bar
            for _ in tqdm(range(size), unit='MB', unit_scale=True, leave=False):
                # Random speeds make it look real
                time.sleep(random.uniform(0.01, 0.1)) 
            print(f"Completed: {name}\n")

        print("All datasets synchronized. Standing by for commands...")
        
        # Keep the script alive so the server stays up
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n[SIGINT] Interrupted by user.")
        print("Cleaning up temporary cache... Done.")
        sys.exit(0)

if __name__ == "__main__":
    start_prank()