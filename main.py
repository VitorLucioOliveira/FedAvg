import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset
from client import generate_client

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

    ## 5. Start Simulation

    ## 6. Save results



if __name__ == '__main__':
    main()

