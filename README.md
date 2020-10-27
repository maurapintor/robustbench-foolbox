# robustbench-foolbox
Foolbox wrapper for robustbench models.

This repository provides the [Robustbench](https://github.com/RobustBench/robustbench) models 
in a Foolbox Native compatible format.

This code requires Foolbox 3.0 or newer.

Example: 
```python
import foolbox as fb
model_url = 'https://github.com/maurapintor/robustbench-foolbox'
model_name = 'Carmon2019Unlabeled'
norm = 'Linf'
fmodel = fb.zoo.get_model(model_url,
                          module_name='load_model', 
                          model_name=model_name, norm=norm)
samples, labels = fb.samples(fmodel, dataset='cifar10', batchsize=10)
_, advs, success = fb.attacks.LinfPGD()(fmodel, samples, labels, epsilons=[8 / 255])
print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
```