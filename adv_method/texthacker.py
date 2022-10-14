import re
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod
import utils as util


class TextHacker(BaseMethod):

    def __init__(self, model, use_gpu = False, device_id = [0], allowed_query_num=2000, neighbor_delta = 5, population_size = 4, local_search_num = 8, synonym_num = 4, embedding_path='', cos_path = ''):

        super(TextHacker,self).__init__(model = model, use_gpu= use_gpu, device_id= device_id)
        self.predictor = self.model.text_pred
        
        self.synonym_num = synonym_num
        self.idx2word, self.word2idx = util.load_embedding_dict_info(embedding_path)
        self.cos_sim = util.load_cos_sim_matrix(cos_path)

        ## Hyper-parameters
        self.allowed_query_num = allowed_query_num
        self.population_size = population_size
        self.neighbor_delta = neighbor_delta
        self.local_search_num = local_search_num
        self.top_k = population_size
        self.base_reward = 0.5
        
        ## Prepare the synonym dict
        self.synonym_dict = {}

        
    def attack(self, x, y):

        log = self.my_attack(x, y, self.allowed_query_num, self.population_size, self.neighbor_delta, self.local_search_num, self.top_k, self.base_reward)

        return log


    def my_attack(self, x, y, allowed_query_num, population_size, neighbor_delta, local_search_num, top_k, base_reward):

        util.setup_seed(2021)
        log ={}
        
        ## Initialize the weight-table and candidate_set
        weight_table, candidate_set, handle_list = self.init_weight_table(x)

        ## Initialize an adversarial example by multip random substitution with the help of weight-table
        init_log, x_adv, x_orig, label = self.adversary_initialization(x, y, allowed_query_num, candidate_set, handle_list)
        log.update(init_log)

        if not log['classification'] or not log['status']:
            return log

        optim_allowrd_query_num = allowed_query_num - log['init_query_number']
        ## Minimize the perturbation between the benign sampel and the adversarial example with hybrid local seach alogrithm
        optim_log = self.perturbation_optimization(x_adv, x_orig, y, optim_allowrd_query_num, population_size, neighbor_delta, local_search_num, top_k, weight_table, candidate_set, base_reward)
        log.update(optim_log)

        return log



    def adversary_initialization(self, x, y, allowed_query_num, candidate_set, handle_list):
        log = {}
        x_orig = x
        raw_len = len(x_orig)

        _, label = self.query_model(x_orig)
        if label != y:
            log['classification'] = False
            return log, None, None, None
        log['classification'] = True

        text = x_orig.copy()
        init_query_number = 1
        x_adv = x_orig.copy()
        while(1):
            ## Substitute all words in x to search the adversarial example
            text = self.replace_word_with_synonym_in_text(text, handle_list, candidate_set)
            _, label = self.query_model(text)
            init_query_number += 1
            ## Update the weight-table
            if label !=y:
                x_adv = text.copy()
                break

            ## Initialization fails
            if init_query_number > allowed_query_num:
                log['status'] = False
                log['init_query_number'] = init_query_number
                return log, text, x_orig, label


        perturbation_rate, perturbation_num = self.count_perturbation_rate(x_orig, x_adv)

        _, label = self.query_model(x_adv)
        log['init_adversarial_example'] = ' '.join(x_adv)
        log['init_perturbation_num'] = perturbation_num
        log['init_perturbation_rate'] = perturbation_rate
        log['init_query_number'] = init_query_number
        if label == y:
            log['status'] = False
        else:
            log['status'] = True

        return log, x_adv, x_orig, label

    


    def perturbation_optimization(self, x_adv, x_orig, y_orig, optim_allowrd_query_num, population_size, neighbor_delta, local_search_num, top_k, weight_table, candidate_set, base_reward):
        log = {}
        best_text = x_adv.copy()
        fitness = self.compute_fitness(x_orig, y_orig, best_text)
        best_fitness = fitness
        optim_query_number = 1


        ## Initialize a initial population
        population_lists = [x_adv]
        fitness_list = [fitness]
        x_adv_before = best_text.copy()
        for m in range(population_size-1):
            ## Utilize the local search to consturct adversaril example
            x_adv, swap_index = self.local_search(x_orig, x_adv_before, neighbor_delta, weight_table, candidate_set)
            if x_adv is None:
                continue
            fitness = self.compute_fitness(x_orig, y_orig, x_adv)
            optim_query_number += 1

            ## Update the weight-table
            weight_table = self.update_weight_table_in_optim(weight_table.copy(), swap_index, fitness, base_reward)

            population_lists.append(list(x_adv))
            fitness_list.append(fitness)
            if fitness == 0:
                continue

            x_adv_before = x_adv.copy()

            ## Update the global optima x_{best}
            if fitness > best_fitness:
                best_text = x_adv.copy()
                best_fitness = fitness
                perturbation_rate, perturbation_num = self.count_perturbation_rate(x_orig, best_text)
                if perturbation_num == 1:
                    _, label = self.query_model(best_text)
                    log['final_adversarial_example'] = ' '.join(best_text)
                    log['optim_perturbation_num'] = perturbation_num
                    log['optim_perturbation_rate'] = perturbation_rate
                    log['optim_query_number'] = 1
                    log['after_attack_label'] = label
                    return log

        ## Perform the hybrid local seach algotithm to minimize the perturbation
        while(1):
            if optim_query_number > optim_allowrd_query_num:
                break
            
            ## Recombination
            l = len(population_lists)
            index = list(range(l))
            np.random.shuffle(index)

            new_population_lists = []
            new_fitness_list = []
            for t in range(len(index) // 2):
                ## Genereate a child sample with the randomly sampled two texts in population
                x_recombination = self.recombination(x_orig, population_lists[index[2*t]], population_lists[index[2*t+1]], weight_table, candidate_set)
                
                ## Record the each sampel and its fitness
                fitness = self.compute_fitness(x_orig, y_orig, x_recombination)
                optim_query_number += 1
                new_population_lists.append(list(x_recombination))
                new_fitness_list.append(fitness)

                if optim_query_number > optim_allowrd_query_num:
                    break
                ## Update the global optima x_{best}
                if fitness > best_fitness:
                    best_text = x_recombination.copy()
                    best_fitness = fitness
                    perturbation_rate, perturbation_num = self.count_perturbation_rate(x_orig, best_text)
                    if perturbation_num == 1:
                        break

            population_lists = population_lists + new_population_lists
            fitness_list = fitness_list + new_fitness_list


            ## Local search
            new_population_lists = []
            new_fitness_list = []
            l = len(population_lists)
            ## Perform multip step local search for each sample in pipulation
            for j in range(l):
                x_adv_before = population_lists[j].copy()
                ## Record the local optima
                local_best_x = x_adv_before.copy()
                local_best_fitness = fitness_list[j]
                
                for m in range(local_search_num):
                    ## Perform the local search to search a sampel form the neighborhood of x_adv_before
                    x_adv, swap_index = self.local_search(x_orig, x_adv_before, neighbor_delta, weight_table, candidate_set)
                    if x_adv is None:
                        break
                    fitness = self.compute_fitness(x_orig, y_orig, x_adv)
                    optim_query_number += 1
                    ## Update the weight-table
                    weight_table = self.update_weight_table_in_optim(weight_table.copy(), swap_index, fitness, base_reward)

                    if optim_query_number > optim_allowrd_query_num:
                        break

                    ## If x_adv is not an adversaril example, we continue to perform the local search on x_adv_before
                    if fitness == 0:
                        continue
                    x_adv_before = x_adv.copy()

                    ## Update the local optima
                    if fitness > local_best_fitness:
                        local_best_x = x_adv.copy()
                        local_best_fitness = fitness

                    ## Update the global optima
                    if fitness > best_fitness:
                        best_text = x_adv.copy()
                        best_fitness = fitness
                        perturbation_rate, perturbation_num = self.count_perturbation_rate(x_orig, best_text)
                        if perturbation_num == 1:
                            break
                    
                new_population_lists.append(local_best_x.copy())
                new_fitness_list.append(local_best_fitness)   


            population_lists = new_population_lists
            fitness_list = new_fitness_list

            ## Select the top S adversarial examples for the next generation
            index = np.argsort(fitness_list)[::-1][:top_k]
            population_lists = np.array(population_lists)[index].tolist()
            fitness_list = np.array(fitness_list)[index].tolist()

            perturbation_rate, perturbation_num = self.count_perturbation_rate(x_orig, best_text)
            if perturbation_num == 1:
                break
            
        
        ## Record the attack performance and return
        perturbation_rate, perturbation_num = self.count_perturbation_rate(x_orig, best_text)
        _, label = self.query_model(best_text)

        confidence_adv, label_adv = self.query_model(best_text)
        confidence_orig, label_orig = self.query_model(x_orig)
        log['adv_confidence'] = confidence_adv
        log['label_adv'] = label_adv
        log['confidence_orig'] = confidence_orig
        log['label_orig'] = label_orig
        
        log['final_adversarial_example'] = ' '.join(best_text)
        log['optim_perturbation_num'] = perturbation_num
        log['optim_perturbation_rate'] = perturbation_rate
        log['optim_query_number'] = optim_query_number

        log['after_attack_label'] = label
        if label == y_orig:
            log['status'] = False
        else:
            log['status'] = True
        
        return log


    def candidate_generate(self, word, sim_score):
        candidate_word_list, candidate_value_list = util.replace_with_synonym(word, 'embedding', self.idx2word, self.word2idx, self.cos_sim)
        candidate_word_list = np.array(candidate_word_list[:self.synonym_num+1])
        candidate_value_list = np.array(candidate_value_list[:self.synonym_num+1])
        index = (candidate_value_list > sim_score)
        candidate_word_list = candidate_word_list[index].tolist()
        candidate_value_list = candidate_value_list[index].tolist()
        return candidate_word_list, candidate_value_list
    
    def init_weight_table(self, x_orig):
        weight_table = []
        candidate_set = []
        handle_list = [-1] * len(x_orig)

        ## Filter the pos, we only operate on adjectives, adverbs, verbs and nouns
        pos_ls = util.get_pos(x_orig)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos:
                    handle_list[i] = 0

        ## Construct the candidate set and initialize the weight for each word
        for i in range(len(x_orig)):
            word = x_orig[i]
            ## Find the synonyms for the word
            if word in self.synonym_dict.keys():
                candidate_list = self.synonym_dict[word]
            else:
                candidate_list, _ = self.candidate_generate(word, 0.5)
                self.synonym_dict[word] = candidate_list
            candidate_set.append(candidate_list)

            l = len(candidate_list)
            w = [0 for _ in range(l)]
            weight_table.append(w)

            ## The word with only one synonym is unoperable
            if(len(candidate_list) <= 1):
                handle_list[i] = -1
        return weight_table, candidate_set, handle_list


    def update_weight_table_in_optim(self, weight_table, swap_index, fitness, base_reward):
        ## Set different reward and update weight-table based on whether the new sample is an adversarial example
        for index, fi, ti in swap_index:
            if fitness == 0:
                weight_table[index][fi] += 2*base_reward
                weight_table[index][ti] -= base_reward
            
            else:
                weight_table[index][fi] -= 2*base_reward
                weight_table[index][ti] += base_reward
        return weight_table


    def replace_word_with_synonym_in_text(self, text, handle_list, candidate_set):
        raw_len = len(text)
        index = [i for i in range(len(handle_list)) if handle_list[i] != -1]

        ## Substitute each word in text with the help of weight-table and record the repalced word
        for i in index:
            text[i] = random.choice(candidate_set[i])

        return text
            
    
    def local_search(self, x_orig, x_new, neighbor_delta, weight_table, candidate_set):
        x_mutate = x_new.copy()

        handle_list = self.generate_handle_list(x_orig, x_mutate)
        index = [i for i in range(len(handle_list)) if handle_list[i] == 1 ]
        
        if len(index) == 0:
            return None, None

         ## Calculate the probability of each word being sampled
        prob = [np.sum(weight_table[i]) for i in range(len(handle_list)) if handle_list[i] == 1 ]
        prob = np.array(prob)
        prob = self.sigmoid(prob)
        prob = [1 - t + 0.0001 for t in prob]
        if np.sum(prob) == 0:
            prob = [p+1 for p in prob] 

        change_ratio, change_num = self.count_perturbation_rate(x_orig, x_new)
        prop = [i+1 for i in range(min(neighbor_delta, change_num)-1)]

        if not prop:
            prop.append(1)
        index_num = min(np.random.choice(prop), len(index))
        index = random.choices(index, prob, k=index_num)
        index = list(set(index))

        ## Replace each sampled word with its candidate word with the help of weight-table
        swap_index = []
        swap_prob = 0.5
        swap_select = np.random.choice([0,1], p = np.array([1-swap_prob, swap_prob]))
        for i in index:
            fi = 0
            for m,w in enumerate(candidate_set[i]):
                if w == x_mutate[i]:
                    fi = m
                    break

            if swap_select == 0:
                ## The word is replaced with the original word with the probability 50%
                x_mutate[i] = x_orig[i]
                swap_index.append([i, fi, 0])
            else:
                ## The word is replaced with its ccandidate word with the probability 50%
                mutate_prob = np.array(weight_table[i])
                mutate_prob = self.sigmoid(mutate_prob)  + 0.01
                ti = random.choices(list(range(len(candidate_set[i]))), mutate_prob)[0]
                swap_index.append([i, fi, ti])
                x_mutate[i] = candidate_set[i][ti]

        return x_mutate, swap_index


    
    def recombination(self, x_orig, x_new_1, x_new_2, weight_table, candidate_set):
        parents = [x_new_1, x_new_2]
        x_recombination = []
        l = len(x_orig)
        for i in range(l):
            ## Select the word from the parents as the word in child according the corresponding weight
            c1, c2 = None, None
            for m,w in enumerate(candidate_set[i]):
                if w == parents[0][i]:
                    c1 = weight_table[i][m]
                if w == parents[1][i]:
                    c2 = weight_table[i][m]
            if c1 and c2:
                prob = [c1,c2]
                prob = np.array(prob)
                prob = self.sigmoid(prob) + 0.01
                prob = np.log(prob + 1)

                index = random.choices([0, 1], prob)[0]
            else:
                index = random.choice([0, 1])
            
            x_recombination.append(parents[index][i])

        return x_recombination


    def count_perturbation_rate(self, x_orig, x_new):
        perturbation_num = 0
        l = len(x_orig)
        for i in range(l):
            if x_orig[i] != x_new[i]:
                perturbation_num += 1
        perturbation_rate = perturbation_num/l
        return perturbation_rate, perturbation_num

    def compute_fitness(self,x_orig, y_orig, x_new):
        _, label = self.query_model(list(x_new))
        if label == y_orig:
            return 0
        else:
            perturbation_rate, perturbation_num = self.count_perturbation_rate(x_orig, x_new)
            return 1-perturbation_rate
    
    
    def generate_handle_list(self, x_orig, x_new):
        handle_list = [1 if x_orig[i] != x_new[i] else 0 for i in range(len(x_orig))]
        return handle_list

    def query_model(self, text):
        probs = self.predictor([text])[0].data.cpu()
        label = torch.argmax(probs, dim=-1).data.numpy()
        return probs, label

        
    def sigmoid(self, x):
        y = 1/(1 + np.exp(-x))
        return y
