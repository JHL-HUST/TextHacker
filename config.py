import time
class Config(object):
    def __init__(self):

        ## GPU params
        self.GPU = dict(
            use_gpu = True,                             # Whether to use GPU
            device_id = [0],                            # The number of the GPU device
        )

        ## Main parameters
        self.CONFIG = dict(
            model_name = 'Bert',                        # Target model name (choices: 'Bert', 'WordCNN', 'WordLSTM')
            attack_name = 'TextHacker',                        # Attack method (choices: 'TextHacker', 'TextHacker_NLI')
        )

        ## Model params
        self.Bert = dict(
            pretrained_dir = './data/model/bert/imdb',  # The path to hold the pretrained BERT
            nclasses = 2,                               # The number of categories of the corresponding dataset
            max_seq_length = 256,                       # Maximum length of text sequence
            batch_size = 32,                            # Batch number in classification
        )
        self.Bert_NLI = dict(
            pretrained_dir = './data/model/bert/imdb',  # The path to hold the pretrained BERT for textual entailment task
            max_seq_length = 128,                       # Maximum length of text sequence
            batch_size = 32,                            # Batch number in classification
        )
        self.WordCNN = dict(
            embedding_path = './data/embedding/glove.6B.200d.txt',                              # The path to hold the word embedding
            nclasses = 2,                               # The number of categories of the corresponding dataset
            batch_size = 32,                            # Batch number in classification
            target_model_path = './data/model/WordCNN/imdb',                                    # The path to hold the pretrained WordCNN
        )

        self.WordLSTM = dict(
            embedding_path = './data/embedding/glove.6B.200d.txt',                              # The path to hold the word embedding
            nclasses = 2,                               # The number of categories of the corresponding dataset
            batch_size = 32,                            # Batch number in classification
            target_model_path = './data/model/WordLSTM/imdb',                                   # The path to hold the pretrained WordLSTM
        )
        

        ## Dataset params
        self.AdvDataset = dict(
            dataset_path = './data/dataset/imdb',       # The path to hold the dataset
        )


        ## Hyper-parameters of attack method params
        self.TextHacker = dict(
            allowed_query_num = 2000,                   # Query budget
            neighbor_delta = 5,                         # Neighborhood size
            population_size = 4,                        # Population size
            local_search_num = 8,                       # Maximum number of local search
            synonym_num = 4,                            # Synonym number
            embedding_path = './data/aux_files/counter-fitted-vectors.txt',       # The path to the counter-fitting embeddings we used to find synonyms
            cos_path =  './data/aux_files/mat.txt'            # The pre-compute the cosine similarity scores based on the counter-fitting embeddings
        )

        self.TextHacker_NLI = dict(
            allowed_query_num = 500,                    # Query budget
            neighbor_delta = 5,                         # Neighborhood size
            population_size = 4,                        # Population size
            local_search_num = 8,                       # Maximum number of local search
            synonym_num = 4,                            # Synonym number
            embedding_path = './data/aux_files/counter-fitted-vectors.txt',       # The path to the counter-fitting embeddings we used to find synonyms
            cos_path =  './data/aux_files/mat.txt'            # The pre-compute the cosine similarity scores based on the counter-fitting embeddings
        )

        ## Log params
        self.Checkpoint = dict(
            log_dir = './log',                          # The log directory where the attack results will be written
            log_filename = '{}_{}_{}.log'.format(self.CONFIG['model_name'], self.CONFIG['attack_name'], time.strftime("%m-%d_%H-%M")),              # The output log filename
        )

        ## Attack setting params
        self.Switch_Method = dict(
            method = 'Batch_Sample_Attack',             # The attack mode: batch attack multiple samples or attack only one sample (choices: 'Batch_Sample_Attack', 'One_Sample_Attack')
        )
        self.Batch_Sample_Attack = dict(
            batch = 1000,                               # The sample number of batch attack
        )
        self.One_Sample_Attack = dict(
            index = 66,                                 # The sample index of attack
        )

    def log_output(self):
        log = {}

        log['CONFIG'] = self.CONFIG
        for name,value in self.CONFIG.items():
            if type(value) is str and hasattr(self,value):
                log[value] = getattr(self,value)
            else:
                log[name] = value
        log['Switch_Method'] = self.Switch_Method['method']
        log['Switch_Method_Value'] = getattr(self, self.Switch_Method['method'])

        return log

    
    def load_parameter(self, parameter):
        for key, value in parameter.items():
            if hasattr(self, key):
                if type(value) is dict:
                    orig_config = getattr(self, key)
                    if orig_config.keys() == value.keys():
                        setattr(self, key, value)
                    else:
                        redundant_key = value.keys() - orig_config.keys()
                        if redundant_key:
                            msg = "there are many redundant keys in config file, e.g.:  " + str(redundant_key)
                            assert None, msg
                        
                        lack_key = orig_config.keys() - value.keys()
                        if lack_key:
                            msg = "there are many lack keys in config file, e.g.:  " + str(lack_key)
                            assert None, msg
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)

        self.Checkpoint['log_filename'] =  '{}_{}_{}.log'.format(self.CONFIG['model_name'], self.CONFIG['attack_name'], time.strftime("%m-%d_%H-%M"))              #log文件名称

        return None
    
    