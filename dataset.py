import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader

# Função para baixar e transformar o dataset MNIST
def get_mnist(data_path: str = "./data"):
    
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    
    return trainset, testset

# Função principal que prepara os dados para os clientes
def prepare_dataset(num_partitions: int, bacth_size: int, val_ratio: float = 0.1):

    trainset, testset = get_mnist() # Pega o dataset completo (60000) [imagem, rótulo]
    
    # IID --> Ceiando lista de clientes e os números de dados em cada uma (iguais)
    
    num_images = len(trainset) // num_partitions # Quantas imagens cada cliente receberá
    partition_len = [num_images] * num_partitions #Lista de clientes com dados vazios
    trainset = random_split(trainset, partition_len, torch.Generator().manual_seed(2023)) # Trainset é agora uma lista de 100 data bases, cada uma com 600 images/rotulos
    
    # create dataloader with train+val support --- data retroactivity FL
 
    trainloaders= []
    valloaders = []
    
    # Separa uma pequena parte dos dados de cada cliente para validação
    for set in trainset:
        num_total = len(set)# Total de dados no cliente [imagem, rotulo]
        num_val = int(val_ratio * num_total)# Separa uma parte (10%) pra validação
        num_train = num_total - num_val # O resto (90%) para testes 
        
        for_train, for_val = random_split(set, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, bacth_size, shuffle=True, num_workers=0)) # Adiciona o DataLoader para a lista de treino
        valloaders.append(DataLoader(for_val, bacth_size, shuffle=False, num_workers=0)) # Adiciona o DataLoader para lista de teste
        
    # Cria um único DataLoader para o teste final
    testlaoders = DataLoader(testset, batch_size=128)
    
    return trainloaders, valloaders, testlaoders