import argparse
import logging
import sys
from pathlib import Path

# Add the repository root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import configuration and training function
import wandb
from mdistiller.engine.cfg import CFG, show_cfg
from train import main

def generate_data_sizes(subset="medium"):
    """Generate data sizes based on the specified subset size"""
    return [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

def generate_learning_rates(subset="medium"):
    """Generate learning rates based on the specified subset size"""
    return [0.0, 0.001, 0.01, 0.1, 0.1778, 0.2, 0.5, 1.0, 2.0]

def train_model():
    """Training function that wandb will call for each hyperparameter combination"""
    # Initialize wandb run and get config
    with wandb.init() as run:
        # Get configuration from wandb
        config = wandb.config
        
        # Update CFG with the parameters from wandb
        CFG.DATASET.SUBSET = config.data_size
        CFG.DA.LR = config.learning_rate
        CFG.EXPERIMENT.NAME = f"resnet32x4_resnet8x4_{config.data_size}_{config.learning_rate}"
        
        # Log the configuration
        logger.info(f"Running with configuration: {dict(config)}")
        
        try:
            # Run training with these hyperparameters
            main(CFG, False, [])
            logger.info(f"Completed run with data_size={config.data_size}, lr={config.learning_rate}")
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            # Even if there's an error, wandb will still track it

def run_wandb_sweep():
    """Set up and run a wandb sweep"""
    parser = argparse.ArgumentParser("WandB Sweep for policy distillation")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--subset", type=str, choices=["small", "medium", "large", "all"], 
                        default="medium", help="Size of parameter sweep")
    parser.add_argument("--prob", type=float, default=1.0, 
                        help="Probability value for DA.PROB")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("--project", type=str, default="cifar_small_data",
                        help="cifar_small_data")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of runs to execute (None = unlimited)")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    global logger
    logger = logging.getLogger("wandb_sweep")
    
    # Load base configuration
    CFG.merge_from_file(args.config)
    
    # Get parameter sets
    data_sizes = generate_data_sizes(args.subset)
    learning_rates = generate_learning_rates(args.subset)
    
    logger.info(f"Setting up sweep with the following parameters:")
    logger.info(f"Data sizes: {data_sizes}")
    logger.info(f"Learning rates: {learning_rates}")
    
    # Define wandb sweep configuration
    sweep_config = {
        "method": "grid",  # Using grid to ensure we try all combinations
        "metric": {
            "name": "best_acc",
            "goal": "maximize"
        },
        "parameters": {
            "data_size": {
                "values": data_sizes
            },
            "learning_rate": {
                "values": learning_rates
            }
        }
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project
    )
    
    logger.info(f"Created sweep with ID: {sweep_id}")
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=train_model, count=args.count)

if __name__ == "__main__":
    run_wandb_sweep()