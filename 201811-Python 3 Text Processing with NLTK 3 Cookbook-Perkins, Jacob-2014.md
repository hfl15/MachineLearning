Latest update 20181206  

This book is good for engineer who want to realize a NLP program. We can learn the structure of NLTK and how to construct comprehensive text processing program on it. Meanwhile, it should be noted that this book place emphasis on basic process or pre-process text data. Exept text classification, this book didn't cover more advanced topic in NLP, like summary, topic model, etc.  
Sum up, this book can be a good reference to preprocess text data, but we should extend other advanced topics in other ways.

# Python 3 Text Processing with NLTK 3 Cookbook", Perkins, Jacob, 2014

1. [Tokenizing Text and WordNet Basics](#01)
2. [Replacing and Correcting Words](#02)
3. [Creating Custom Corpora](#03)
4. [Part-of-speech Tagging](#04)
5. [Extracting Chunks](#05)
6. [Transforming Chunks and Trees](#06)
7. [Text Classification](#07)
8. [Distributed Processing and Handling Large Datasets](#08)
9. [Parsing Specific Data Types](#09)
    
<h2 id="01"> 1. Tokenizing Text and WordNet Basics </h2>

This chapter will cover the basics of tokenizing text and using WordNet. 

**Tokenization** is a method of breaking up a piece of text into many pieces, such as sentences and words.  
**Wordnet** is a dictionary designed for programmatic access by natural language processing systems. It has many different use cases, including: (1) Looking up the definition of a word, (2) Finding synonyms and antonyms, (3) Exploring word relations and similarity, (4) Word sense disambiguation for words that have multiple uses and definitions. 

<h3> Tokenizing </h3>

**Tokenizing text into sentences**  
You can use function `sent_tokenize` from the `nltk.tokenize` module. `sent_tokenize` function uses an instance of `PunktSentenceTokenizer` from the `nltk.tokenize.punkt` module. This instance has already been trained and works well for many European languages.  

**Tokenizing sentences into words**  
You can use function `word_tokenize` from the `nltk.tokenize`. `word_tokenize` function is a wrapper function that calls `tokenize` on an instance of the `TreebankWordTokenizer` class. It works by separating words using spaces and punctuation, but does not discard the punctuation.   
There are more tokenizer, e.g. `WhitespaceTokenizer`, `SpaceTokenizer`, `PunktWordTokenizer`, `WordPunctTokenizer`, `TreebankWordTokenizer`.

**Tokenizing sentences using regular expressions**  
Regular expressions can be used if you want complete control over how to tokenize text. As regular expressions can get complicated very quickly, author only recommend using them if the word tokenizers covered in the previous recipe are unacceptable.  
There are two way to realize the above procedure: (1) create an instance of `RegexpTokenizer`, (2) use `regexp_tokenize` function from the `nltk.tokenize` module.

**Training a sentence tokenizer**  
NLTK's default sentence tokenizer is general purpose, and usually works quite well. But sometimes it is not the best choice for your text. In such cases, training your own sentence tokenizer can result in much more accurate sentence tokenization.   
The `PunktSentenceTokenizer` class uses an unsupervised learning algorithm to learn what constitutes a sentence break.  
The specific technique used in this case is called **sentence boundary detection** and it works by counting punctuation and tokens that commonly end a sentence, such as a period or newline, then using the resulting frequencies to decide what the sentence boundaries should actually look like.  
**Most of the time, the default sentence tokenizer will be sufficient. This is covered in the first recipe, Tokenizing text into sentences.**

**Filtering stopwords in a tokenized sentence**  
**Stopwords** are common words that generally do not contribute to the meaning of a sentence, at least for the purposes of information retrieval and natural language processing. These are words such as _the_ and _a_. Most search engines will filter out stopwords from search queries and documents in order to save space in their index.   
You can use function `stopwords` in the `nltk.corpus` module.   
You can see the list of all English stopwords uding `stopwords.words('english')` or by examining the word list file at `nltk_data/corpora/stopwords/english`. There are also stopword lists for many other languages. You can see the complete list of languages using the `fileids` method as `stopwords.fileids()`.    
If you'd like to create your own `stopwords` corpus, see the `Creating a wordlist corpus` recipe in Chapter 3.

<h4> WordNet Basics </h4>


**Looking up Synsets for a word in WordNet**  
**Synset** instances, which are groupings of synonymous words that express the same concept. Many words have only one Synset, but some have several. In this recipe, we'll explore a single Synset, and in the next recipe, we'll look at several in more detail.  
Specifically, you can use `wordnet` in `nltk.corpus` then looking up any word in WordNet using `wordnet.synsets(word)` to get a list of Synsets. There are more, you can work with hypernyms, look up a simplified part-of-speech tag as `syn.pos()`.

**Looking up lemmas and synonyms in WordNet**  
Building on the previous recipe, we can also look up lemmas in WordNet to find synonyms of a word. A **lemma** (in linguistics), is the canonical form or morphological form of a word.  
There's more, (1) Many words have multiple Synsets because the word can have different meanings depending on the context, you can **get all possible synonyms**. (2) Some lemmas also have **antonyms**.

**Calculating WordNet Synset similarity**  
Synsets are organized in a _hypernym_ tree. This tree can be used for reasoning about the similarity between the Synsets it contains. The closer the two Synsets are in the tree, the more similar they are. (1) **Wu-Palmer Similarity**, which is a scoring method based on how similar the word senses are and where the Synsets occur relative to each other in the hypernym tree. (2) **Path and Leacock Chordorow (LCH) similarity**.  

**Discovering word collocations**  
**Collocations** are two or more words that tend to appear frequently together. As with many aspects of natural language processing, context is very important. And **for collocations, context is everything**!  
In the case of collocations, the context will be a document in the form of a list of words. Discovering collocations in this list of words means that we'll find common phrases that occur frequently throught the text.   
For optimization, we can apply a **frequency filter** to filter out low frequent words. There are two main hyperparameters **Scoring functions** and **Scoring ngrams**.
    
<h2 id="02"> 2. Replacing and Correcting Words </h2>

In this chapter, we will go over various word replacement and correction techniques. The recipes cover the gamut of linguistic compression, spelling correction, and text normalization. All of these methods can be very useful for preprocessing text before search indexing, document classification, and text analysis.

**Stemming words**  
**Stemming** is a technique to remove affixes from a word, ending up with the stem. For example, the stem of `cooking` is `cook`. It is note that the result stem is not always a valid word, for example the stem of `cookery` is `cookeri`.   
One of the most commom stemming algorithm is the **Porter stemming algorithm** by Martin Porter, which has realized in NLTK and you can use `PorterStemmer` in `nltk.stem` module.  
There are more stemmer, e.g. `RegexpStemmer`, `SnowballStemmer`, `LancasterStemmer`.
    
**Lemmatizing words with WordNet**  
**Lemmatization** is very similar to stemming, but is more akin to synonym replacement. A lemma is a root word, as opposed to the root stem. So **unlike stemming, you always left with a valid word that means the same thing**. However, the word you end up with can be completely different.   
Specifically, you can use `WordNetLemmatizer` from the `nltk.stem` module. The `WordNetLemmatizer` class is a thin wrapper around the `wordnet` corpus and uses the `morphy()` function of the `WordNetCorpusReader` class to find a lemma.   
There's more, you can combining stemming with lemmatization to compress words more than either process can by itself. These cases are somewhat rare, but they do exist:   
>\>\>\> stemmer.stem('buses')  
'buse'  
\>\>\> lemmatizer.lemmatize('buses')  
'bus'  
\>\>\> stemmer.stem('bus')  
'bu'   

That is nearly a 60% compression rate!. 

**Replacing words matching regular expressions**  
Now, we are going to get into the process of replacing words. If stemming and lemmatization are a kind of linguistic compression, then **word replacement can be thought of as error correction or text normalization**.  
How to do it: (1) define a number of replacement patterns. This will be a list of tuple pairs, where the first element is the pattern to match with and the second element is the repacement. (2) create a `RegexpReplacer` class that will compile the patterns and provide a `replace()` method to substitute all the found patterns with their replacements.   
There's more: (1) The `RegexpReplacer` class can take any list of replacement patterns for whatever purpose. (2) You can apply this procedure, replacement, before tokenization.

**Removing repeating characters**  
In everyday language, people are often not strictly grammatical. They will write things such as `I looooooove it` in order to emphasize the word `love`. However, computers don't know that `looooooove` is a variation of `love` unless they are told. This recipe presents a method to remove these annoying repeating characters in order to end up with a `proper` English word.  
A **backreference** is a way to refer to sa previously mached group in a regular  expression. This will allow us to match and remove repeating characters.   
Specifically, you can use `RepeatReplacer` class in `replacers`. The `RepeatReplacer` class starts by compiling a regular expression to match and define a replacement string with backreferences. 

**Spelling correction with Enchant**  
Replacing repeating characters is actually an extreme form of spelling correction. This recipe take on the less extreme case of correcting minor spelling issues using **Echant** — a spelling correction API.   
You can use or create a new class called `SpellingReplacer` in `replace.py`.   
The `SpellingReplacer`  class starts by creating a reference to an Enchant dictionary. Then, in the `replace()` method, it first checks whether the given word is present in the dictionary. If the word is not found, it looks up a list of suggestions and returns the first suggestion, as long as its edit distance is less than or equal to `max_dist`.  
You can use language dictionaries other than `en`, such as `en_GB`, assuming the dictionary has already been installed. What's more, Enchant also support **personal word lists**.

**Replacing synonyms**  
It is often useful to reduce the vocabulary of a text by replacing words with common synonyms.   
First, you can create a `WordReplacer` class in `replacers.py` that takes a word replacement mapping. The `WordReplacer` class is simply a class wrapper around a Python dictionary. The `replace()` method looks up the given word in its `word_map` dictionary and returns the replacement synonym if it exists.   
It should be noted that Hardcoding synonyms in a Python dictionary is not a good long-term solution. Two better alternatives are to store the synonyms in a CSV file or in a YAML file. Choose whichever format is easiest for those who maintain your synonym vocabulary. 

**Replacing negations with antonyms**  
The opposite of synonym replacement is **antonym replacement**. An **antonym** is a word that has the opposite meaning of another word.   
You can create an `AntonymReplacer` class in `replacers.py`. The `AntonymReplacer` class has two methods: `replace()` and `replace_negations()`. The `replace()` method takes a single word and an optional part-of-speech tag, then looks up the Synsets for the word in WordNet. 
    
<h2 id="03"> 3. Creating Custom Corpora </h2>

If you want to train your own model, such as part-of-speech tagger or text classifier, you will need to create a custom corpus to train on.

**Setting up a custom corpus**  
A **corpus** is a collection of text documents, and **corpora** is the plural of corpus. This comes from the Latin word for body; in this case, a body of text. So a **custom corpus** is really just a bunch of text files in a directory, often alongside many other directories of text files.   
NLTK defines a list of data directories, or paths, in `nltk.data.path`. Our custom corpora must be within one of these paths so it can be found by NLTK.   
Finally, author create a subdirectory in corpora to hold their custom corpus. Let's call it `cookbook`, giving us the full path, which is `~/nltk_data/corpora/cookbook`.  
What's more, be sure to use choose unambiguous names for your files so as not to conflict with any existing NLTK data. 
    
**Creating a wordlist corpus**  
The `WordListCorpusReader` class is one of the simplest `CorpusReader` classes. It provides access to a file containing a list of words, one word per line.  
The `stopwords` corpus is a good example of a multifile `WordListCorpusReader`. 

**Creating a part-of-speech tagged word corpus**  
You can use `TaggedCorpusReader` from `nltk.corpus.reader`.`TaggedCorpusReader` has following functions, `words()`, `sents()`, `paras()`, `tagged_words()`, `tagged_sents()`, `tagged_paras()`.   
The `TaggedCorpusReader` class tries to have good defaults, but you can customize them by passing in your own tokenizers at the time of initialization. You can customize the word tokenizer, the sentence tokenizer, the paragraph block reader, the tag separator.   
NLTK 3.0 provides a method for converting known tagsets to a universal tagset. A **tagset** is just a list of part-of-speech tags used by one or more corpora. The **universal tagset** is a simplified and condensed tagset composed only 12 part-of-speech tags. 
 
**Creating a chunked phrase corpus**  
A **chunk** is a short phrase within a sentence. You can use `ChunkedCorpusReader` from `nltk.corpus.reader`.

**Creating a categorized text corpus**  
If you have a large corpus of text, you might want to categorize it into separate sections. This can be helpful for organization, or for text classification.  
The easiest way to categorize a corpus is to have one file for each category.    
You can categorize corpus with category file, categorized tagged corpus reader, or categorized corpora.
  
**Creating a categorized chunk corpus reader**  
NLTK provides a `CategorizedPlaintextCorpusReader` and `CategorizedTaggedCorpusReader` class, but there's no categorized corpus reader for chunked corpora. So this recipe going to make one.  

**Lazy corpus loading**  
Loading a corpus reader can be an expensive operation due to the number of files, file sizes, and various initialization tasks.  
To speed up module import time when a corpus reader is defined, NLTK provides a `LazyCorpusLoader` class that can transform itself into your actual corpus reader as soon as you need it. This way, you can define a corpus reader in a common module without it slowing down module loading.   
The `LazyCorpusLoader` class stores all the arguments given, but otherwise does nothing until you try to access an attribute or method.

**Creating a custom corpus view**  
 A **corpus view** is a class wrapper around a corpus file that reads in blocks of tokens as needed. Its purpose is to provide a view into a file without reading the whole file at once (since corpus files can often be quite large).

**Creating a MongoDB-backed corpus reader**  
All the corpus readers we've dealt with so far have been file-based. That is in part due to the design of the `CorpusReader` base class, and also the assumption that most corpus data will be in text files.  
However, sometimes you'll have a bunch of data stored in a database that you want to access and use just like a text file corpus.   
This recipe cover the case where you have documents in MongoDB, and you want to use a particular field of each document as your block of text.   
MongoDB is a document-oriented database that has become a popular alternative to relational databases such as MySQL.

**Corpus editing with file locking**  
Corpus readers and views are all read-only, but there will be times when you want to add to or edit the corpus files. However, modifying a corpus file while other processes are using it, such as through a corpus reader, can lead to dangerous undefined behavior. This is where file locking comes in handy.  
Here are two file editing functions: `append_line()` and `remove_line()`.

<h2 id="04"> 4. Part-of-speech Tagging </h2>

**Part-of-speech(POS)** tagging is the process of converting sentence, in the form of a list of words, into a list of tuples, where each tuple is of the form (**word**, **tag**). The **tag** is a part-of-speech tag, and signifies whether the word is a noun, adjective, verb, and so on.   
POS is a necessary step before **chunking**. You can also use POS for **grammar analysis** and **word sense disambiguation**.  
All taggers in NLTK are in the `nltk.tag` package and inherit from the `TaggerI` base class. 

**Default tagging**  
`DefaultTagger` class provides a baseline to measure accuracy improvements.   
The `DefaultTagger` class takes a single argument, the tag you want to apply. We'll give it NN, which is the tag for a singular noun. `DefaultTagger` is most useful when you choose the most common part-of-speech tag. Since nouns tend to be the most common types of words, a noun tag is recommended.  
There's more, you also can tag sentences and untag a tagged sentence.  

**Training a unigram part-of-speech tagger**  
A **unigram** generally refers to a single token. Therefore, a unigram tagger only uses a single word as its context for determining the part-of-speech tag.  
Here, you can set a **minimum frequency cutoff threshold** to filter out the low frequent tags in context.  

**Combining taggers with backoff tagging**  
**Backoff tagging** is one of the core features of `SequentialBackoffTagger`. It allows you to chain taggers together so that if one tagger doesn't know how to tag a word, it can pass the word on to the next backoff tagger. If that one can't do it, it can pass the word on to the next backoff tagger, and so on until  there are no backoff taggers left to check.   
If your final backoff tagger is `DefaultTagger`, `None` will never be returned.

**Training and combining ngram taggers**  
An **ngram** is a subsequence of _n_ items.   
In addition to `UnigramTagger`, there are two more `NgramTagger` subclasses: `BigramTagger` and `TrigramTagger`. These two taggers are good at handling words whose part-of-speech tag is context-dependent.    
In the case of this book, `BigramTagger` and `TrigramTagger` can make a contribution is when they were combined with backoff tagging.   
The `backoff_tagger` function from the `tag_util` module creates an instance of each tagger class in the list, giving it `train_sents` and the previous tagger ass a backoff. The order of the list of tagger classes is quite important: the first class in list (`UnigramTagger`) will be trained first and given the initial backoff tagger (the `DefaultTagger`). This tagger will then become the backoff tagger for the next tagger class in the list. The final tagger returned will be an instance of the last tagger class in the list (`TrigramTagger`).   
The author also try quadgram tagger, but it's slightly worse than before, when they stopped with the `TrigramTagger`. So, the lesson is that too much context can have a negative effect on accuracy. 

**Creating a model of likely word tags**  
To find the most common words, we can use `nltk.probability.FreqDist` to count word frequencies in the `treebank` corpus. Then, we can create a `ConditionalFreqDist` class for tagged words, where we count the frequency of every tag for every word.

**Tagging with regular expressions**  
You can use regular expression matching to tag words.  
For this recipe to make sense, you should be familiar with the regular expression syntax and Python's `re` module.  
The `RegexpTagger` class expects a list of two tuples, where the first element in the tuple is a regular expression and the second element is the tag. 

**Affix tagging**  
The `AffixTagger` class is another `ContextTagger` subclass, but this time the context is either the prefix or the suffix of a word. This means the `AffixTagger` class is able to learn tags based on fixed-length substrings of the beginning or ending of a word.  
The `AffixTagger` class also take a `min_stem_length` keyword argument, with a default value of 2, this parameter control the word length to learn. If the word length is less than `min_stem_length` plus the absolute value of `affix_length`, then `None` is returned by the `context()` method. 

**Training a Brill tagger**  
The `BrillTagger` class is a transformation-based tagger. It is the first tagger that is not a subclass of `SequentialBackoffTagger`. Instead, the `BrillTagger` class uses a series of rules to correct the results of an initial tagger. These rules are scored based on how many errors they correct minus the number of new errors they produce.

**Training the TnT tagger**  
**TnT** stands for **Trigrams'n'Tags**. It is a statistical tagger based on second order Markov models.   
The TnT tagger has a slightly different API than the previous taggers we've encountered. You must explicitly call `train()` method after you've created it.   
It is noted that training is fairly quick, but tagging is significantly slower than the other taggers we've covered. This is due to all the floating point math that mush be done to calculate the tag probabilities of each word.   
There's more, (1) author recommend always passing `Trained=True` if you also pass `unk` tagger. (2) The parameter `N` can control the number of possible solutions the tagger maintains while trying to guess the tags for a sentence. You can select a best `N` by beam search. (3) You can pass `C=True` to the TnT constructor if you want capitalization of words to be significant.

**Using WordNet for tagging**  
It's a very restricted set of possible tags, and many words have multiple Synsets with different part-of-speech tags, but this information can be useful for tagging unkonw words. WordNet is essemtially a giant dictionary, and it's likely to contain many words that are not in your training data.

**Tagging proper names**  
Using the included `names` corpus, we can create a simple tagger for tagging names as proper nouns.

**Classifier-based tagging**  
The `ClassifierBasedPOSTagger` class uses classification to do part-of-speech tagging.   
You pass in training sentences, it trains an internal classifier, and you get a very accurate tagger.  
Notice a slight modification to initialization: `train_sents` must be passed in as the `train` keyword argument.  
It defaults to training a `NaiveBayesClassifier` class with the given training data. Once this classifier is trained, it is used to classify word features produced by the `feature_detector()` method.  
The `ClassifierBasedTagger` class is often the most accurate tagger, but it's also one of the slowest taggers. If speed is an issue, you should stick with `BrillTagger` class based on a backoff chain of `NgramTagger` subclasses and other simple taggers.  
You also can use `MaxentClassifier`, but it still slower than `NaiveBayesClassifier`.

**Training a tagger with NLTK-Trainer**  
There are many different ways to train taggers, and it's impossible to know which methods and parameters will work best without doing training experiments. But training experiments can be tedious, since they often involve many small code changes (and lots of cut and paste) before you converge on an optimal tagger. In an effort to simplify the process, and make my own work easier, the author created a project called **NLTK-Trainer.**  
**NLTK-Trainer** is a collection of scripts that give you the ability to run training experiments without writing a single line of code. 

<h2 id="05"> 5. Extracting Chunks </h2>

**Chunk extraction**, or **partial parsing**, is the process of extracting short phrases from a part-of-speech tagged sentence. This is different from full parsing in that we're interested in standalone **chunks**, or **phrases**, instead of full parse trees. The idea is that meaningful phrases can be extracted from a sentence by looking for particular patterns of part-of-speech tags.

**Chunking and chinking with regular expressions**  
Using modified regular expressions, we can define **chunk patterns**. These are patterns of part-of-speech tags that define what kinds of words make up a chunk. We can also define patterns for what kinds of words should not be in a chunk. These unchunked words are known as **chinks**.  
How you create and combine patterns is really up to you. Pattern creation is a process of trial and error, and entirely depends on what your data looks like and which patterns are easiest to express.

**Merging and splitting chunks with regular expressions**  
A `MergeRule` class can merge two chunks together based on the end of the first chunk and the beginning of the second chunk. A `SplitRule` class will split a chunk into two chunks based on the specified split pattern.

**Expanding and removing chunks with regular expressions**  
`ExpandLeftRule`: Add unchunked (chink) words to the left of a chunk.  
`ExpandRightRule`: Add unchunked (chink) words to the right of a chunk.  
`UnChunkRule`: Unchunk any matching chunk.

**Partial parsing with regular expressions**  
So far, we've only been parsing noun phrases. But `RegexpParser` supports grammars with multiple phrase types, such as verb phrases and prepositional phrases. We can put the rules we've learned to use and define a grammar that can be evaluated against the `conll2000` corpus, which has `NP`, `VP`, and `PP` phrases. 

**Training a tagger-based chunker**  
The experiment result is pretty darn accurate! Training a chunker is clearly a great alternative to manually specified grammas and regular expressions. 

**Classification-based chunking**  
Unlike most part-of-speech taggers, the `ClassifierBasedTagger` class learns from features. That means we can create a `ClassifierChunker` class that can learn from both the words and part-of-speech tags, instead of only the part-of-speech tags as the `TagChunker` class does.

**Extracting named entities**  
**Named entity recognition** is a specific kind of chunk extraction that uses entity tags instead of, or in addition to, chunk tags. Common entity tags include `PERSON`, `ORGANIZATION`, and `LOCATION`. Part-of-speech tagged sentences are parsed into chunk trees as with normal chunking, but the labels of the trees can be entity tags instead of chunk phrase tags.

**Extracting proper noun chunks**  
A simple way to do named entity extraction is to chunk all proper nouns (tagged with `NNP`). We can tag these chunks as `NAME`, since the definition of a proper noun is the name of a person, place, or thing.

**Extracting location chunks**  
To identify `LOCATION` chunks, we can make a different kind of `ChunkParserI` subclass that uses the `gazetteers` corpus to identify location words. The `gazetteers` corpus is a `WordListCorpusReader` class that contains the following location words: Country names, U.S. states and abbreviations, Major U.S. cities, Canadian provinces, Mexican states.  
The `LocationChunker` class starts by constructing a set of all locations in the `gazetteers` corpus.

**Training a named entity chunker**  
We can train our own named entity chunker using the `ieer` corpus, which stands for **Information Extraction: Entity Recognition**. It takes a bit of extra work, though, because the `ieer` corpus has chunk trees but no part-of-speech tags for words.  
Despite the non-ideal training data, the `ieer` corpus provides a good place to start for training a named entity chunker. The data comes from `New York Times` and `AP Newswire` reports. Each doc from `ieer.parsed_docs()` also contains a headline attribute that is a `Tree`.

**Training a chunker with NLTK-Trainer**

    
<h2 id="06">6. Transforming Chunks and Trees </h2>

The **chunk transforms** are for grammatical correction and rearranging phrases without loss of meaning. The **tree transforms** give you ways to modify and flatten deep parse trees.   
What's important in this chapter is what you can do with a chunk, not where it came from.
 
**Filtering insignificant words from a sentence**  
This recipe will introduce how to remove the insignificant words and keep the significant ones by looking at their part-of-speech tags. 
Filtering insignificant words can be a good complement to stopword filtering for purposes such as search engine indexing and querying and text classification. 

**Correcting verb forms**  
It's fairly common to find incorrect verb forms in real-world language. You can correct these mistakes by creating verb correction mappings that are used depending on whether there's a plural or singular noun in the chunk.  

**Swapping verb phrases**  
Swapping the words around a verb can eliminate the passive voice from particular phrases. This kind of normalization can also help with frequency analysis, by counting two apparently different phrases as the same phrases.

**Swapping noun cardinals**  
In a chunk, a cardinal word, tagged as CD, refers to a number, such as 10. These cardinals often occur before or after a noun. For normalization purposes, it can be useful to always put the cardinal before the noun.

**Swapping infinitive phrases**  
An infinitive phrase has the form `A of B`, such as `book of recipes`. These can often be transformed into a new form while retaining the same meaning, such as `recipes book`. 

**Singularizing plural nouns**  
As we saw in the previous recipe, the transformation process can result in phrases such as `recipes book`. Thiss is a NNS followed by a NN, when a more proper version of the phrase would be `recipe book`, which is a NN followed by another NN. We can do another transform to correct these improper plural nouns. 

**Chaining chunk transformations**  
The transform functions defined in the previous recipes can be chained together to normalize chunks. The resulting chunks are often shorter with no loss of meaning. 

**Converting chunk tree to text**  
At some point, you may want to convert a `Tree` or subtree back to a sentence or  chunk string. This is mostly straightforward, except when it comes to properly outputting punctuation.

**Flattening a deep tree**  
Some of the included corpora contain parsed sentences, which are often deep trees of nested phrases. Unfortunately, these trees are too deep to use for training a chunker, since IOB tag parsing is not designed for nested chunks. To make these usable for chunker training, we must flatten them.

**Creating a shallow tree**  
Previous recipe flattened a deep `Tree` by only keeping the lowest level subtrees. In this recipe, we'll keep only the highest level subtrees instead.

**Converting tree labels**  
Parse trees often have a variety of `Tree` label types that are not present in chunk trees. If you want to use parse trees to train a chunker, then you'll probably want to reduce this variety by converting some of these tree labels to more common label types.

 
<h2 id="07">7. Text Classification </h2>

**Text classification** is a way to categorize documents or pieces of text.  
A **binary classifier** decides between two labels, such as positive or negative.  
**Multi-label classifier** can assign one or more labels to a piece of text.   
  
**Bag of words feature extraction**   
**Text feature extraction** is the process of transforming what is essentially aa list of words into a feature set that is usable by a classifier.  
The NLTK classifiers expect `dict` style feature sets, so you must therefore transform text into a dict.  
You also can filter stopwords or include significant bigrams. 

**Training a Naive Bayes classifier**  
The `NaiveBayesClassifier` class has two methods that are quite useful for learning about your data. Both methods take a keyword argument `n` to control how many results to show.   
The `most_informative_features()` method returns a list of the form [(feature name, feature value)] ordered by most informative to least informative.   
The `show_most_informative_features()` method will print out the results from `most_informative_features()`.   

**Training a decision tree classifier**  
The `DecisionTreeClassifier` class can take much longer to train the `NaiveBayesClassifier` class. For that reason, author has overridden the default parameters so it trains faster. These parameters will be explained later. 

**Training a maximum entropy classifier**  
The `MaxentClassifier` class algorithms can be quite memory hungry, so you may want to quit all your other programs while training a `MaxentClassifier` class, just to be safe. 

**Training scikit-learn classifiers**  
This recipe didn't access `scikit-learn` modelss directly in this recipe. Instead, author use NLTK's `SKlearnClassifier` class, which is a wapper class around a `sciket-learn` model to make it conform to NLTK's `ClassifierI` interface. This means that the `SklearnClassifier` class can be trained and used much like the classifiers we've used in the previous recipes in this chapter.

**Measuring precision and recall of a classifier**

**Calculating high information words**  
A **high information** word is a word that is strongly biased towards a single classification label.   
The **low information** words are words that are common to all labels.   
The reason this works is that using only high information words reduces the noise and confusion of a classifier's internal model. If all the words/features are highly biased one way or the other, it's much easier for the classifier to make a correct guess. 

**Combining classifiers with voting**

**Classifying with multiple binary classifiers**

**Training a classifier with NLTK-Trainer**

<h2 id="08">8. Distributed Processing and Handling Large Datasets </h2>

NLTK is great for in-memory, single-processor natural language processing. This chapter cover two case: (1) **execnet** was used to do parallel and distributed processing with NLTK. (2) Introduce how to use the **Redis** data structure sever/database to store frequency distributions and more. 

**Distributed tagging with execnet**

**Distributed chunking with execnet**

**Parallel list processing with execnet**

**Storing a frequency distribution in Redis**

**Storing a conditional frequency distribution in Redis**

**Storing an ordered dictionary in Redis**

**Distributed word scoring with Redis and execnet**

<h2 id="09"> 9. Parsing Specific Data Types </h2>
  
**Parsing dates and times with dateutil**  

**Timezone lookup and conversion**

**Extracting URLs from HTML with lxml**

**Cleaning and stripping HTML**

**Converting HTML entities with BeautifulSoup**

**Detecting and converting character encodings**