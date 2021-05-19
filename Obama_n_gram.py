import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.autograd import Variable

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2,5) #2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

#Collect data

CONTEXT_SIZE = 3
EMBEDDING_DIM = 10

test_sentence = []
for i in range(2): #436
    if (i == 8 or i == 14):
        print("")
    else:
        name = "raw_speeches/speech" + str(i) + ".txt"
        data = open(name, "r")
        test_sentence += data.read().split()

#print(test_sentence)

'''
test = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
'''

trigrams = [([test_sentence[i], test_sentence[i+1], test_sentence[i+2]], test_sentence[i+3])
            for i in range(len(test_sentence) - 3)]

print(trigrams[:3])

vocab = set(test_sentence) #find unique words
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

print("about to train!")
for epoch in tqdm.trange(10):
    total_loss = 0
    for context, target in trigrams:

        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype = torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)

        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype = torch.long))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)
print(losses)

#print(model.embeddings.weight[word_to_ix["American"]])

words = [word_to_ix["personal"], word_to_ix["political"], word_to_ix["interests"]]
lookup_tensor = torch.tensor(words, dtype = torch.long)
print(lookup_tensor)
log_probs = model.forward(lookup_tensor)
#arg_max = torch.argmax(log_probs)
#print("Log probs: ")
#print(log_probs)
#print(arg_max)
topten = torch.topk(log_probs, 10)
print("\n top ten \n")
topten
print (topten)
#print(word_to_ix[arg_max])
for i in topten:
    print(i)
    word = [key for key, i in word_to_ix.items()]
    print(word)
#word = [key for key, value in word_to_ix.items() if value == arg_max]

#print(word)

words = [word_to_ix["have"], word_to_ix["a"], word_to_ix["solemn"]]
lookup_tensor = torch.tensor(words, dtype = torch.long)
log_probs = model.forward(lookup_tensor)
arg_max = torch.argmax(log_probs)
print(arg_max)
#print(word_to_ix[arg_max])
word = [key for key, value in word_to_ix.items() if value == arg_max]

print(word)
