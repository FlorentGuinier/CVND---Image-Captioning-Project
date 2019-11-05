import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        #from `sparse word indices` to embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        #LSTM taking embed_size either from picture or a word.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        #From embedded space to vocab (ie probabilities * one hot)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # The goal here -> For each input (image[feature]/caption[seqLenth]) pair
        # we want to predict predictedCaption[x+1] from caption[x]. And we do that in batch (indexed via dim 0).
        batch_size = features.size()[0]
        
        shouldLogSize = False
        if (shouldLogSize):
            print('features',features.size())
            print('captions',captions.size())
            print('batch_size', batch_size)
        
        ############ CAPTION
        #remove last token as it is <end> we don't want to predict from it. 
        captions = captions[:, :-1]
        #captions (batch, seqLength-1) -> embeddingVec(batch, seqLength-1, embed_size)
        embeddingVec = self.embedding(captions)
        if (shouldLogSize):
            print('embeddingVec', embeddingVec.size())
        
        ############ FEATURE
        #features (batch, embed_size) -> reshape -> features (batch, 1, embed_size)
        features = features.view(batch_size, 1, -1)
        if (shouldLogSize):
            print('features', features.size())
        
        ############ INPUT TO LSTM (images + sequenced captions)
        #features (batch, 1, embed_size) + embeddingVec(batch, seqLength-1, embed_size) -> x (batch, seqLength, embed_size)
        x = torch.cat((features, embeddingVec), 1)
        if (shouldLogSize):
            print('x before LSTM', x.size())
        
        ############ PREDICTION
        #(batch, seqLength, hiddenstate) -> LSTM -> (batch, seqLength, hiddenstate)
        x, (self.hidden, self.cell) = self.lstm(x, None) #no need to expose hidden/cell state it is only a property of each sequence.
        if (shouldLogSize):
            print('x after LSTM', x.size())
        
        ############ BACK TO WORDS
        #(batch, seqLength, hiddenstate) -> Dense -> (batch, seqLength, vocab_size)
        x = self.fc(x) 
        if (shouldLogSize):
            print('x after Dense', x.size())

        #output is expected to be of size [batch_size, captions.shape[1], vocab_size]
        return x;
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # The goal here -> We are provided the first `embedding` ie the image feature and the maximum caption size to produce.
        # We will generate the caption iteratively until <end> token is reach or max_len is reach.
        
        end_word = 1 #TODO would be cleaner to get it from data_loader.dataset.vocab.end_word
        captionIndices = []
        
        for i in range(max_len):
            inputs, states = self.lstm(inputs, states)     #(1, 1, hiddenstate)
            inputs = self.fc(inputs)                       #(1, 1, vocab_size)
            vocabIdx = inputs.argmax()                     #(1)
            captionIndices.append(vocabIdx.item())
            if (vocabIdx == end_word):
                return captionIndices
            inputs = self.embedding(vocabIdx).view(1,1,-1) #(1, 1, hiddenstate)
        
        return captionIndices
        