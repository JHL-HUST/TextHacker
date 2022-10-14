import utils as util

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import boto3


class AWS(nn.Module):
    def __init__(self, config = None):
        super(AWS, self).__init__()
        self.comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')
        # self.label_dict = {'NEGATIVE': 0, 'POSITIVE': 1, 'NEUTRAL': 2}

    def text_pred(self, text):
        outs = []
        for x in text:
            input_text = ' '.join(x)
            re = self.comprehend.detect_sentiment(Text=input_text, LanguageCode='en')
            output = F.softmax(torch.tensor([re['SentimentScore']['Negative'], re['SentimentScore']['Positive']])).tolist()
            outs.append(output)

        outs = torch.tensor(outs)
        return outs


def load_AWS(config):
    model = AWS(config=config)
    return model