import adv_method
import model_loader
import user as user_module
import config as config_module
import utils as utils_module
import os
import argparse
import yaml

def main():
    utils_module.setup_seed(2021)

    ## Required parameters for target model and hyper-parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        default=None,
                        help="The config parameter yaml file contains the parameters of the dataset, target model and attack method",
                        type=str)

    parser.add_argument('--vGPU',
                        nargs='+',
                        type=int,
                        default=None,
                        help="Specify which GPUs to use.")

    args = parser.parse_args()


    ## Universal parameters
    config = config_module.Config()

    if args.config:
        assert os.path.exists(args.config), "There's no '" + args.config + "' file."
        with open(args.config, "r") as load_f:
            config_parameter = yaml.load(load_f)
            config.load_parameter(config_parameter)

    if args.vGPU:
        config.GPU['device_id'] = args.vGPU

    ## Save the parameters into log
    Log = config.log_output()

    ## Configure the GPU
    use_gpu = config.GPU['use_gpu']
    if use_gpu:
        device = config.GPU['device_id']
        os.environ["CUDA_VISIBLE_DEVICES"]= str(device[0])

    ## Prepare the dataset
    texts, labels = utils_module.read_corpus(config.AdvDataset['dataset_path'], csvf=False)

    ## Prepare the target model
    model = getattr(model_loader, 'load_' + config.CONFIG['model_name'])(**getattr(config, config.CONFIG['model_name']), config = config)

    ## Prepare the attack method
    attack_parameter = getattr(config, config.CONFIG['attack_name'])
    attack_name = config.CONFIG['attack_name']
    attack_method = getattr(adv_method, attack_name)(model, **config.GPU, **attack_parameter)

    ## Prepare the attacker
    attacker = user_module.Attacker(model, config, attack_method)
    

    ## Start the attack
    dataloader = list(zip(texts, labels))
    log = attacker.start_attack(dataloader)
    Log.update(log)

    
    ## Save and print the Log
    print(config.Checkpoint['log_dir'], config.Checkpoint['log_filename'])
    utils_module.ensure_dir(config.Checkpoint['log_dir'])
    filename = os.path.join(config.Checkpoint['log_dir'], config.Checkpoint['log_filename'])
    f = open(filename,'w')

    for key, value in Log.items():
        if 'print' not in key:
            print('    {:15s}: {}'.format(str(key), value))
        log = {}
        log[key] = value
        utils_module.log_write(f, log)



if __name__ == "__main__":
    main()
