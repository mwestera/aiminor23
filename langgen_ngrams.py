import random
import pandas as pd

"""
Python script to read twitter data, extract n-gram continuation probabilities, and start generating language based on a prompt.

TODOs (not all of which will necessarily be done on Monday):
[x] - The are some avoidable inefficiencies in the n-grams function that can be greatly sped-up; this will be explained 
  in class. For now, don't apply it to a big dataset :)
[x] - Let it take prompt from user input
[x] - Generalize to n-grams for any (?) n
[x] - If an n-gram isn't found, resort to an (n-1)-gram, etc...
- Separate reading/processing the corpus ('model training') from generating output ('model deployment'), probably
  best in two separate scripts. (Because the former takes quite a while for a million tweets!)
- Document our functions
- Split off punctuation
- (Always:) Refactoring (i.e., restructuring the code, improving readability, SoC, DRY, etc.)

Clarifying the purpose of this code:
- This code was created entirely during class, for the purpose of showing some natural language processing ideas and some programming principles 'in action' 
  while keeping the code itself accessible for students just starting with Python.
- The purpose of this code is NOT to show how a professional Python programmer would approach this, which
  would involve more existing libraries, perhaps more 'object-oriented programming', and ultimately a more
  sophisticated model. We _may_ see a bit more of that on Monday.
  
Try it yourself!
- This is an ordinary .py file, not an iPython/Jupyter notebook. So if you know only how to run a notebook, you should 
  either 1. expand your knowledge :) or 2. copy this code piece by piece into a notebook and it will likely work.
- I'm not allowed to share twitter data with you. So if you want to run this code, you need to create your own
  twitter corpus, or use a different corpus.
- In the latter case, you may have to adapt the read_corpus function (or create a new one). If you use a single
  plain text file as input, this could be as simple as 
  
  words = open('my_text_file.txt').read().lower().split()
  
  Consider using the following file: 
  https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt .
  
  Simply save it to the same folder as your .py file or notebook; but if you run your notebook 
  'in the cloud' then it may be a bit different (a web search might help).
"""

def main():
    # prompt = input('Enter a prompt:')
    # prompt_n = int(input('Enter a desired n:'))
    prompt = 'covid is'
    prompt_n = 3

    words = read_corpus('uk_tbcov.csv')
    words = [word.lower() for word in words]    # list comprehension syntax
    big_huge_probability_mapping = {}
    for m in range(1, prompt_n + 1):
        probability_mapping = extract_continuation_probabilities(words, n=m)
        big_huge_probability_mapping.update(probability_mapping)

    generate_words(prompt, big_huge_probability_mapping, n=prompt_n)


def read_corpus(path):
    df = pd.read_csv(path, nrows=10000)
    tweets = list(df['text'])
    words = []
    for tweet in tweets:
        tweet_words = tweet.split()
        words.extend(tweet_words)
    return words


# Separation of concerns (SoC), do not repeat yourself (DRY), encapsulation

def sample_next_word(probability_distribution):
    candidates = list(probability_distribution.keys())
    probabilities = list(probability_distribution.values())
    choice = random.choices(candidates, weights=probabilities)[0]
    return choice


def extract_continuation_probabilities(words, n=3):

    prefix_to_ngrams = {}
    for i in range(len(words) - (n - 1)):
        ngram = tuple(words[i:i+n])
        prefix = ngram[:(n-1)]
        if prefix not in prefix_to_ngrams:
            prefix_to_ngrams[prefix] = []
        prefix_to_ngrams[prefix].append(ngram)

    big_counts_dictionary = {}
    for prefix, ngrams in prefix_to_ngrams.items():
        continuations = [ngram[-1] for ngram in ngrams]
        counts = {word: 0 for word in continuations}
        for word in continuations:
            counts[word] += 1
        big_counts_dictionary[prefix] = counts

    return big_counts_dictionary


def generate_words(prompt, probability_mapping, n):

    n_generated_words = 0
    all_text = prompt

    print(prompt, ' (prompt)')

    while n_generated_words < 1000:
        n_generated_words += 1

        words = all_text.split()

        for m in range(n, 0, -1):
            try:
                prompt_as_tuple = tuple(words[-(m-1):]) if m > 1 else tuple()
                probability_distribution_for_next_word = probability_mapping[prompt_as_tuple]
            except KeyError as e:
                continue

            next_word = sample_next_word(probability_distribution_for_next_word)
            all_text = all_text + ' ' + next_word

    print(all_text)


if __name__ == '__main__':
    main()