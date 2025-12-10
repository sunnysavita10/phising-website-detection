import os
def print_header(message):
    print("\n" + "="*len(message))
    print(message)
    print("="*len(message) + "\n")
    
def ensure_artifacts_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")