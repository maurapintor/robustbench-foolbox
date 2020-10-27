import foolbox as fb

from load_model import load_model_to_device


def create(model_name, norm, device='cpu'):
    model = load_model_to_device(model_name=model_name, norm=norm, device=device)
    model.eval()

    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))

    return fmodel
