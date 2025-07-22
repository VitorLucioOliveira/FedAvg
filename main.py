import hydra
from omegaconf import DictConfig, OmegaConf

import flwr as fl
from dataset import prepare_dataset
from client import generate_client
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    ## 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    
    ## 2. Prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)
    
    print(len(trainloaders), len(trainloaders[0].dataset))
    
    ## 3. Define your clients
    client = generate_client(trainloaders, validationloaders, cfg.num_classes )

    ## 4. Define strategy
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_roud_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn= get_on_fit_config(cfg.config_fit),
                                         evaluate_fn= get_evaluate_fn(cfg.num_classes, testloader))
    
    ## 5. Start Simulation

    ## 6. Save results



if __name__ == '__main__':
    main()

