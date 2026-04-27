import math
import torch
import matplotlib.pyplot as plot
import os

# Preparing words
with open('rockyou.txt', encoding='utf-8', errors='ignore') as list:
    words = list.read()

words = words.split('\n')

# We'd need to retrieve a list of characters that we have in our dataset
chars = sorted(set(''.join(words))) #  we have 709 characters for the full corpus
chars.append('~')

# We will now need to init a n x n tensor and calculate the count of each probability, where n = chars + 1
bigram_set = torch.ones([len(chars), len(chars)], dtype=float)

# Here's a python dict mapping of index to char and vice versa
char_to_index = {}
index_to_char = {}

for index, char in enumerate(chars):
    char_to_index[char] = index
    index_to_char[index] = char

if not os.path.exists('tensor.pt'):
    print(f"loaded in {len(chars)} characters and {len(words)} words. Training model now!")
    # The way we need to think about this is that p(a, b) is the probability of b following a. 
    # This part can definitely be optimised. Way too slow for wordlists with multimillion words.
    counter = 0
    for word in words:
        word = '~' + word + '~'
        counter += 1
        for x, y in zip(word, word[1:]):
            x_index = char_to_index[x]
            y_index = char_to_index[y]

            bigram_set[x_index, y_index] += 1
        if counter % 10000 == 0:
            print(f"{counter} words completed")

    print("Training complete!")
    torch.save(bigram_set, 'tensor.pt')

else:
    print("Found trained model in your local directory at /tensor.pt")
    bigram_set = torch.load('tensor.pt')

# Now we need to normalise these values and sample using torch operations
summed_set = torch.sum(bigram_set, 1, keepdim=True)
bigram_set /= summed_set

# heat map of trained dataset
# plot.imshow(bigram_set.numpy(), cmap='Blues')
# plot.colorbar()
# plot.show()

g = torch.Generator(device='cpu').manual_seed(18891)

def generate_weak_password():
    counter = char_to_index['~']
    password = []
    while True:
        chosen_char_index = torch.multinomial(bigram_set[counter], 1, True, generator=g).item()
        chosen_char = index_to_char[chosen_char_index]
        password.append(chosen_char)
        counter = chosen_char_index
        if chosen_char == '~':
            break


def score_your_password(w: str):
    w = '~' + w + "~"
    nll = 0
    for char1, char2 in zip(w, w[1:]):
        char1_index = char_to_index[char1]
        char2_index = char_to_index[char2]
        probability = bigram_set[char1_index][char2_index]
        nll -= torch.log(probability)
    return(nll)

to_score = input("what password would you like to score? ")
print(score_your_password(to_score))
