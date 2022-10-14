import os
import json
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader

from config import *
import utils as util
import pickle
import datetime
import time
import language_tool_python


class Attacker(object):

    def __init__(self, model, config, attack_method):
        self.model = model
        self.config = config
        self.attack_method = attack_method

        self.use_gpu = False
        self.device_ids = [0]
        self.device = torch.device('cpu')
        if self.config.GPU['use_gpu']:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                self.device_ids = self.config.GPU['device_id']
                assert len(self.device_ids) > 0,message
                self.device = torch.device('cuda', self.device_ids[0])
                self.use_gpu = True



        self.attack_name = self.config.CONFIG['attack_name']
        self.tool = language_tool_python.LanguageTool('en-US')



    def start_attack(self, dataloader):
        attack_method = self.config.Switch_Method['method']
        log = {}
        if attack_method == 'One_Sample_Attack':
            index = getattr(self.config, attack_method)['index']
            for i,(text, label) in enumerate(dataloader):
                if (i == index):
                    log = self.one_sample_attack(text, label)
                    break
        elif attack_method == 'Batch_Sample_Attack':
            log = self.batch_sample_attack(dataloader, **getattr(self.config, attack_method))

        return log



    def one_sample_attack(self, text, label):
        log = {}
        attack_log = self.attack_method.attack(text, label)
        log['pre_data'] = ' '.join(text)
        log['pre_label'] = label
        log.update(attack_log)
        return log

    def batch_sample_attack(self, data_loader, batch):  
        log = {}
        use = util.USE('./data/aux_files')

        ## Record the attack performance
        success = 0                             # The sample number of successful attacks
        classify_true = 0                       # The sample number of successfully classified after attacks
        sample_num = 0                          # The total number of samples

        query_number_list = []                  # The query number of each attack for target model
        perturbation_rate_list = []             # The perturbation rate of the adversarial example after each attack
        process_time_list = []                  # The processing time for each attack
        
        sim_list = []                           # The semantic similarity between the benign sample and the adverasrial sample after each attack
        orign_grammer_error_list = []           # The grammatical error rate of the benign sample
        adv_grammer_error_list = []             # The grammatical error rate of the adversaril example


        ## Record the attack performance without the 25% perturation rate limit
        unlimit_success = 0                                           
        unlimit_perturbation_rate_list = []                 
        unlimit_sim_list = []                    

        unlimit_orign_grammer_error_list = []
        unlimit_adv_grammer_error_list = []


        for i,(x, y) in enumerate(data_loader):
            if i == batch:
                break
            
            starttime = datetime.datetime.now()
            one_log = self.one_sample_attack(x, y)
            endtime = datetime.datetime.now()
            process_time =  (endtime - starttime).seconds
            process_time_list.append(process_time)


            if not one_log['classification']:
                message = 'The {:3}-th sample is not correctly classified'.format(i)
                log['print_{}'.format(i)] = message
                print(message)
                continue

            sample_num += 1

            # Record the query number
            init_query_number = 0
            init_query_number = one_log['init_query_number']
            optim_query_number = 0
            if 'optim_query_number' in one_log.keys():
                optim_query_number = one_log['optim_query_number']
            query_time = init_query_number + optim_query_number
            query_number_list.append(query_time)


            if(one_log['status']):
                unlimit_success += 1

                ## Record the perturbation rate
                perturbation_rate = one_log['optim_perturbation_rate']
                unlimit_perturbation_rate_list.append(perturbation_rate)
                
                ## Calculate the semantic similarity and grammatical error rate
                adv_text = one_log['final_adversarial_example']
                sim = use.semantic_sim([' '.join(x)], [adv_text])[0][0]
                unlimit_sim_list.append(sim)

                orign_grammer_error = len(self.tool.check(' '.join(x)))/len(x)
                adv_grammer_error = len(self.tool.check(adv_text))/len(x)
                unlimit_orign_grammer_error_list.append(orign_grammer_error)
                unlimit_adv_grammer_error_list.append(adv_grammer_error)
                
                ## The attack would be considered as success only if the perturbation rate of the adversarial example is smaller than 25%.
                if perturbation_rate > 0.25:
                    message = 'The {:3}-th sample takes {:3}s, with the perturbation rate: {:.5}, semantic similarity: {:.5}, query number: {:4}, which exceeds the 25% perturbation rate limit. Attack fails.'.format(i, process_time, perturbation_rate, sim, optim_query_number)
                    print(message)
                    log['print_{}'.format(i)] = message
                    continue
                
                ## Record the attack performance
                success += 1
                perturbation_rate_list.append(perturbation_rate)
                sim_list.append(sim)
                adv_grammer_error_list.append(adv_grammer_error)
                orign_grammer_error_list.append(orign_grammer_error)

                message = 'The {:3}-th sample takes {:3}s, with the perturbation rate: {:.5}, semantic similarity: {:.5}, query number: {:4}. Attack succeeds.'.format(i, process_time, perturbation_rate, sim, optim_query_number)
                print(message)
            else:
                classify_true += 1
                message = 'The {:3}-th sample takes {:3}s, Attack fails'.format(i, process_time)
                print(message)
            
            log['print_{}'.format(i)] = message
        
        message = '\nA total of {:4} samples were selected, {:3} samples were correctly classified, {:3} samples were attacked successfully and {:4} samples failed'.format(batch, sample_num, success, sample_num - success)
        print(message)
        log['print_last'] = message


        acc = sample_num/batch                                                      # The classification accuracy of target model
        attack_acc = (classify_true + unlimit_success - success)/batch              # The classification accuracy of target model after attack
        success_rate = success/sample_num                                           # The attack success rate of attack method
        average_perturbation_rate = np.mean(perturbation_rate_list).item()          # The average perturbation rate of the adversarial example
        average_sim = np.mean(sim_list).item()                                      # The average semantic similarity of the adversarial example


        unlimit_after_attack_acc = classify_true/batch                          
        unlimit_success_rate = unlimit_success/sample_num                             
        unlimit_average_perturbation_rate = np.mean(unlimit_perturbation_rate_list).item()                  
        unlimit_average_sim = np.mean(unlimit_sim_list).item()
        

        average_query_number = np.mean(query_number_list).item()                    # The average query number of each attack
        average_process_time = np.mean(process_time_list).item()                    # The average process time of each attack
        

        log['before_attack_acc'] = acc
        log['unlimit_after_attack_acc'] = unlimit_after_attack_acc
        log['unlimit_success_rate'] = unlimit_success_rate
        log['unlimit_mean_perturbation_rate'] = unlimit_average_perturbation_rate
        log['unlimit_mean_sim'] = unlimit_average_sim
        log['unlimit_grammer_error_increase'] = np.mean(unlimit_adv_grammer_error_list) - np.mean(unlimit_orign_grammer_error_list)


        log['after_attack_acc'] = attack_acc
        log['success_rate'] = success_rate
        log['mean_perturbation_rate'] = average_perturbation_rate
        log['mean_sim'] = average_sim
        log['grammer_error_increase'] = np.mean(adv_grammer_error_list) - np.mean(orign_grammer_error_list)

        log['mean_query_number'] = average_query_number
        log['mean_process_time'] = average_process_time


        return log

        