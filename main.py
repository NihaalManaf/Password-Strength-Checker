import os
import torch

# Preparing words
with open('rockyou.txt', encoding='utf-8', errors='ignore') as list:
    words = list.read()

words = words.split('\n')[:20] # We have over million words in our list so for development, lets use first 20 for now

# We'd need to retrieve a list of characters that we have in our dataset
chars = set(''.join(words)) #  we have 709 characters for the full corpus




# We will now need to init a n x n tensor and calculate the count of each probability, where n = chars + 1
bigram_set = torch.ones([len(chars) + 1, len(chars) + 1], dtype=float)
print(bigram_set.shape)

# Here's a python dict mapping of index to char and vice versa
char_to_index = {}
index_to_char = {}

for index, char in enumerate(chars):
    char_to_index[char] = index
    index_to_char[index] = char

print(char_to_index)


# The way we need to think about this is that p(a, b) is the probability of b following a. 
for word in words:
    word = "<E>" + word + "<E>"
    for x, y in zip(word, word[1:]):
        x_index = char_to_index[x]
        y_index = char_to_index[y]

        bigram_set[x_index, y_index] += 1

# Now we need to normalise these values and sample using torch operations
summed_set = torch.sum(bigram_set, 0, keepdim=True)
bigram_set /= summed_set

focus = 0
