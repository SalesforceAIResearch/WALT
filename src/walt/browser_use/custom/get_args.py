import argparse
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    # env setup
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--platform", type=str, default="MacOS")
    # eval setup
    parser.add_argument("--max_steps", type=int, default=15)
    
    # agent setup
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "bedrock"])
    parser.add_argument("--model", type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_planner", action="store_true")
    parser.add_argument("--planner_model", type=str, default="o3-mini-2025-01-31")
    parser.add_argument("--planner_temperature", type=float, default=0.0)
    parser.add_argument("--local_kb_path", type=str, default="./kb")
    # query rephraser related
    parser.add_argument("--query_rephraser_model", type=str, default="gpt-5-mini-mini-2024-07-18")
    # retriever related
    parser.add_argument("--retriever_model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--retriever_cache_dir", type=str, default="./cache")
    parser.add_argument("--retriever_top_k", type=int, default=1)
    parser.add_argument("--retriever_threshold", type=float, default=0.3)
    # narrative memory summarizer related
    parser.add_argument("--narrative_memory_summarizer_model", type=str, default="gpt-4.1-2025-04-14")
    # browser agent related
    parser.add_argument("--browser_agent_module", type=str, default="AgentWithCustomPlanner")
    parser.add_argument("--browser_agent_model", type=str, default="gpt-4.1-2025-04-14")
    # test setup
    parser.add_argument("--purpose", type=str, default="inital_test")
    parser.add_argument("--splits", type=str, default="configuration")

    # logging related
    parser.add_argument("--result_dir", type=str, default="outputs")
    parser.add_argument("--generate_gif", action="store_true")
    
    
    
    
    
    

    

    
    args = parser.parse_args()
    return args