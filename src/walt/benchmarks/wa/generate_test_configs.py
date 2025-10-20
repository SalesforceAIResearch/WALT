"""Generate WebArena test data by replacing URL placeholders with environment variables"""
import os
import json
from dotenv import load_dotenv

# Load environment variables  
load_dotenv()

def main() -> None:
    # Get URLs from environment variables
    GITLAB = os.environ.get("GITLAB", "http://localhost:8023")
    REDDIT = os.environ.get("REDDIT", "http://localhost:9999")
    SHOPPING = os.environ.get("SHOPPING", "http://localhost:7770") 
    SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "http://localhost:7780/admin")
    WIKIPEDIA = os.environ.get("WIKIPEDIA", "http://localhost:8888")
    MAP = os.environ.get("MAP", "http://localhost:3000")
    
    print("Generating WebArena test configs...")
    print(f"GITLAB: {GITLAB}")
    print(f"REDDIT: {REDDIT}")
    print(f"SHOPPING: {SHOPPING}")
    print(f"SHOPPING_ADMIN: {SHOPPING_ADMIN}")
    print(f"WIKIPEDIA: {WIKIPEDIA}")
    print(f"MAP: {MAP}")
    
    if not os.path.exists("test_configs/test.raw.json"):
        print("Error: Raw config file not found: test_configs/test.raw.json")
        return
    
    with open("test_configs/test.raw.json", "r") as f:
        raw = f.read()
        
    # Replace URL placeholders with environment variables
    raw = raw.replace("__GITLAB__", GITLAB)
    raw = raw.replace("__REDDIT__", REDDIT)
    raw = raw.replace("__SHOPPING__", SHOPPING)
    raw = raw.replace("__SHOPPING_ADMIN__", SHOPPING_ADMIN)
    raw = raw.replace("__WIKIPEDIA__", WIKIPEDIA)
    raw = raw.replace("__MAP__", MAP)
    
    # Write the processed config file
    with open("test_configs/test.json", "w") as f:
        f.write(raw)
    
    # Create individual test files
    os.makedirs("test_configs/webarena", exist_ok=True)
    data = json.loads(raw)
    for idx, item in enumerate(data):
        with open(f"test_configs/webarena/{idx}.json", "w") as f:
            json.dump(item, f, indent=2)
    
    print(f"Generated {len(data)} test configs in test_configs/webarena/")
    print("Test data generation complete!")


if __name__ == "__main__":
    main()
