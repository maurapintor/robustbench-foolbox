import logging
import os

import torch
from robustbench.utils import rm_substr_from_state_dict, download_gdrive


def load_model_to_device(model_name, model_dir='./models', norm='Linf', device='cpu'):
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("Device set to 'cuda', but cuda is not available. "
                        "Switching to 'cpu'...")
        device = 'cpu'
    from robustbench.model_zoo.models import model_dicts as all_models
    model_dir += '/{}'.format(norm)
    model_path = '{}/{}.pt'.format(model_dir, model_name)
    model_dicts = all_models[norm]
    if not isinstance(model_dicts[model_name]['gdrive_id'], list):
        model = model_dicts[model_name]['model']()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.isfile(model_path):
            download_gdrive(model_dicts[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=device)

        # needed for the model of `Carmon2019Unlabeled`
        try:
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')

        model.load_state_dict(state_dict)
        if device == 'cuda':
            model.cuda()
        model.eval()
        return model

    # If we have an ensemble of models (e.g., Chen2020Adversarial)
    else:
        model = model_dicts[model_name]['model']()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for i, gid in enumerate(model_dicts[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i), map_location=device)
            try:
                state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
            except:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            model.models[i].load_state_dict(state_dict)
            if device == 'cuda':
                model.models[i].cuda()
            model.eval()
        if device == 'cuda':
            model.cuda()
        model.eval()
        return model
