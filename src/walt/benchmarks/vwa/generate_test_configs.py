"""Generate test configs by replacing URL placeholders with environment variables"""
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main() -> None:
    # Get URLs from environment variables
    DATASET = os.environ.get("DATASET", "visualwebarena")
    CLASSIFIEDS = os.environ.get("CLASSIFIEDS", "http://localhost:9980")
    REDDIT = os.environ.get("REDDIT", "http://localhost:9999") 
    SHOPPING = os.environ.get("SHOPPING", "http://localhost:7770")
    WIKIPEDIA = os.environ.get("WIKIPEDIA", "http://localhost:8888")
    
    if DATASET == "visualwebarena":
        print("Generating VisualWebArena test configs...")
        print(f"CLASSIFIEDS: {CLASSIFIEDS}")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        
        inp_paths = [
            "test_configs/visualwebarena/test_classifieds_v2.raw.json",
            "test_configs/visualwebarena/test_shopping_v2.raw.json", 
            "test_configs/visualwebarena/test_reddit_v2.raw.json",
        ]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__CLASSIFIEDS__": CLASSIFIEDS,
        }
    else:
        raise ValueError(f"Dataset not supported: {DATASET}")
        
    for inp_path in inp_paths:
        if not os.path.exists(inp_path):
            print(f"Warning: Raw config file not found: {inp_path}")
            continue
            
        output_dir = inp_path.replace('.raw.json', '')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(inp_path, "r") as f:
            raw = f.read()
        
        # Replace URL placeholders with environment variables
        for k, v in replace_map.items():
            raw = raw.replace(k, v)

        # Write the processed config file
        with open(inp_path.replace(".raw", ""), "w") as f:
            f.write(raw)
            
        # Generate individual test files
        data = json.loads(raw)
        for idx, item in enumerate(data):
            with open(os.path.join(output_dir, f"{idx}.json"), "w") as f:
                json.dump(item, f, indent=2)
                
        print(f"Generated {len(data)} test configs in {output_dir}/")
    
    print("Test config generation complete!")


if __name__ == "__main__":
    main()