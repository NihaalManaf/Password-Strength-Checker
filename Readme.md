## TLDR

This is a simple password strength checker that works by calculating the negative log loss on a password based on a bigram model trained on a popular password wordlist, `rockyou.txt` used commonly in penetration testing.

## Motivation

This is probably not a good idea because the probability of an upcoming character is always super dependent on all the characters before that and not just the previous character. This is just an opportunity for me to build a bigram model.

## How it works

A bigram model heavily depends on the corpus its trained on and there exists vast amounts of password wordlists of commonly used passwords and so there could exist some sort of learning a bigram model can model and the NLL from the characters of a password could be indicative of how common the password could be! Lots of coulds in the previous sentence. 

All I did here was train a bigram over `rockyou.txt` which is a popular pen testing wordlist that comes preinstalled in kali linux. Then I ask for your password, from then I'd iterate through and look up the probability of each char c appearing after the character behind it and take the negated log loss!

The higher your score here, the 'better' your password. Maximise your loss fn for this case haha

### Proof it works

A quick sanity check across a mix of weak, common, and stronger passwords (higher = better):

| Password                     | Score  |
|------------------------------|--------|
| `123456`                     | 16.21  |
| `qwerty`                     | 21.33  |
| `hunter2`                    | 22.55  |
| `nihaal`                     | 23.00  |
| `iloveyou`                   | 25.20  |
| `password`                   | 26.63  |
| `aaaaaa`                     | 27.94  |
| `Tr0ub4dor&3`                | 56.51  |
| `correcthorsebatterystaple`  | 73.02  |
| `xK7#mQ9vL2pZ`               | 90.85  |
| `j8Hq@2Wm^5Rt*Yp`            | 110.95 |
| `Zx9$Qw!7Vb#3Np&Lk`          | 120.56 |

The higher the better. I know proof by example isn't a thing but we can incline towards believing this is a simple but solid strat and it's only O(n), where n is the length of characters in the password we're testing. 

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
