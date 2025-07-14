from collections import OrderedDict
from typing import Dict
from flwr.common import  Scalar, NDArrays

import flwr as fl
import torch
from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes) -> None:
        super().__init__()
        
        self.trainloader = trainloader
        self.valloader = valloader
        
        # Modelo a ser treinado
        self.model = Net(num_classes)
        
        # Dispositivo a ser treinado
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)
        
    def get_parameters(self, config: Dict[str, Scalar]):
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # update do cliente com os parametros do servidor central
        
        self.set_parameters(parameters)
        
        lr = config['lr']
        momentum = config['momentum'] # pra que serve?
        epochs = config['local_epochs']
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        # fazer o treino local
        train(self.model, self.trainloader, optimizer)
        
        return self.get_parameters(), len(self.trainloader.dataset), {} #tirar os dados por privacidade (len)
        
        
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Testar o modelo global no cliente local
        self.set_parameters(parameters)
        
        loss, accuracy = test(self.model, self.valloader, self.device)
        
        return float(loss), len(self.valloader), {'accuracy': accuracy}
    
    
def generate_client(trainloaders,  valloadres, num_classes):
    
    def client(cid: str):
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloadres[int(cid)],
                            num_classes=num_classes)

    return client