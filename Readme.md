This is a simple password strength checker that works by calculating the negative log loss on a password based on a bigram model trained on a popular password wordlist, `rockyou.txt` used commonly in penetration testing.

## Motivation

This is probably not a good idea because the probability of an upcoming character is always super dependnat on the all characters before that and not just the previous characters. This is just an opportunity for me to build a bigram model.

## How it works

A bigram model heavily depends on the corpus its trained on and there exists vast amounts of password wordlists of commonly used passwords and so there could exist some sort of learning a bigram model can model and the NLL from the characters of a password could be indidcative of how common the password could be! Lots of coulds in the previous sentence. 

All I did here was train a bigram over `rockyou.txt` which is a popular pen testing wordlist that comes preinstalled in kali linux. Then I ask for your password, from then I'd iterate through and look up the probabiltiy of the

## How `tensor.pt` is handled 

Training over the full `rockyou.txt` corpus is slow, so the trained count tensor is cached to disk:

- On first run, the script trains the bigram counts and saves them to `tensor.pt` via `torch.save`.
- On subsequent runs, if `tensor.pt` exists in the working directory, it is loaded directly with `torch.load` and training is skipped.
- To retrain from scratch (e.g. after changing the wordlist), delete `tensor.pt` and run again.

Note: the saved tensor is tied to the character ordering produced from the current `rockyou.txt`. If you swap the wordlist, delete the cache so the indices don't get out of sync.

//AI written

## How to run

1. Make sure `rockyou.txt` is in the same directory as `main.py` (unzip `rockyou.txt.zip` if needed).
2. Install dependencies:
   ```bash
   pip install torch matplotlib
   ```
3. Run the script:
   ```bash
   python main.py
   ```
4. When prompted, enter the password you want to score. The script will print its negative log likelihood under the bigram model.

//AI Written