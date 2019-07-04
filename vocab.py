import os
import random
from collections import Counter
from nltk.corpus import brown, stopwords
from collections import defaultdict

if __name__ == '__main__':
    """ Generate train/test vocabs for chart labels. """
    
    try:
        words = brown.tagged_words()
    except:
        import nltk
        nltk.download('brown')
        words = brown.tagged_words()

    nouns = []

    for (word, pos) in words:
        if pos == 'NN':
            nouns.append(word)

    counter = Counter(nouns)
    vocab = [noun for (noun, _) in counter.items()]
    random.shuffle(vocab)
    train_vocab = vocab[:len(vocab)//2]
    test_vocab = vocab[len(vocab)//2:]

    print(f'Numbers of train_vocab: {len(train_vocab)}')
    print(f'Numbers of test_vocab: {len(test_vocab)}')
    
    if not os.path.exists('resources'):
        os.mkdir('resources')
    
    with open('resources/train_vocab.txt', 'w') as file:
        for word in train_vocab:
            print(word, file=file)
    
    with open('resources/test_vocab.txt', 'w') as file:
        for word in test_vocab:
            print(word, file=file)