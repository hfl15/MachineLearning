# Sentiment Analysis and Opinion Mining

Liu, Bing. "Sentiment analysis and opinion mining." Synthesis lectures on human language technologies 5.1 (2012): 1-167. Google cite: 3662.

This book is a good start to get into Sentiment Analysis field, in which concepts and tasks are constructed in structuring so that reader can have a systematic and comprehensive cognition. What's more author review a banch of typical papers, which can be the lure to specific task in sentiment analysis. But it should be note that, this book was published in year 2012, it is a long time up to now, year 2018, in the meanwhile there are various great changes can't be neglected in NLP field, for example the most significant breakthrough of Deep Learning which has spread to various domains. So it is not enough to understand the sentiment analysis comprehensively just by this book, it is recommend to review latest materials after that.     

**Sentiment analysis** also called **opinion mining**, is the field of study that analyzes people's _opinion_, _sentiments_, _evaluations_, _appraisals_, _attitudes_, and _emotions_ towards entities such as _products_, _services_, _organizations_, _individuals_, _issues_, _events_, _topics_, and _their attributes._   

Sentiment Analysis became popular around the year 2000, because (1) a wide range of application, (2) many challenge research problems, and (3) a huge volume of opinionated data were available.   

In formal, quintuple `$ (a_{ij}, e_{i}, s_{ijkl}, h_{k}, t_{l}) $` denote a sentiment `$s_{ijkl}$` on aspect/attribute `$a_{ij}$` of entity `$e_{i}$` from holder `$h_{k}$` in time `$t_{l}$`. Sentiment analysis can be viewed as a process to distill all of sentiments denoted by above quintuple, which can be divided into 6 main task, (1) entity extraction and categorization, (2) aspect extraction and categorization, (3) opinion holder extraction and standardization, (4) time extraction and standardization, (5) aspect sentiment classification (e.g. 5 sentiment ratings,emotional negative(-2), rational negative(-1), neutral(0), rational positive(+1), and emotional positive(+2).) (6) opinion quintuple generation.  

In general the opinion/sentiment can divide into _regular opinion_ and _comparative opinion_, where regular opinion can further divide into _direct opinion_ and _indirect opinion_ and the previous take more focus currently.   
Sentiment analysis has been investigated mainly at three levels: (1) document level (detail in chapter 3), (2) sentence level (detail in chapter 4), (3) entity and aspect level (detail in chapter 5).  

It is note that the degree of difficulty to sentiment analysis is different among various genres, for example Twitter postings (tweets) are short (at most 140 characters) and informal, Reviews is easier to analysis because they are highly focused with little irrelevant information, forum discussion are perhaps the hardest to deal with because the users there can discuss anything and also interact with one another.   
In terms of the degree of difficulty, there is also the dimension of different application domains. Opinions about products and services are usually easier to analyze. Social and political discussions are much harder due to complex topic and sentiment expressions, sarcasms and ironies.

The model of sentiment analysis can be divided into supervised and unsupervised. From the perspective supervised machine learning, we can address problem with regression or classification model, which depend on type of label is numeric or not. But it is a challenge problem to collect enough labeled training data, to meet this problem transfer learning and unsupervised learning are taken into consideration. And most of unsupervised sentiment analysis process rely on a sentiment lexicon so the 6th chapter go deeper in the topic of Sentiment Lexicon Generation. 

This book also go deep into some important and challenge problem in sentiment analysis, (1) Chapter 7, Opinion Summarization, introduce how to construct and visualize sentiment result in a systematic form, (2)  Chapter 8, Analysis of Comparative Opinions, emphasize the important and challenge of comparative opinions execpt the regular opiniont introduced in previous chapters, (3) Chapter 9, Opinion Search and Retrieval, remind that it is important to support opinion search or retriveal by entity/aspect or  person/organization like traditional search engine. (4) Chapter 10, Opinion Spam Detection, with a view to the security problem in sentiment analysis field, and point out that it is important to identify every fake review, fake reviewer, and fake reviewer group. (5) Chapter 11, Quality of Reviews, this topic is related to opinion spam detection, but is also different because low quality reviews may not be spam or fake reviews, and fake reviews may not be perceived as low quality reviews by readers. The objective of this task is to determine the quality, helpfulness, usefulness, or utility of each review.

# Content

- [x] [1. Sentiment Analysis: A Fascinating Problem](#01) 
- [x] [2. The Problem of Sentiment Analysis](#02) 
- [x] [3. Document Sentiment Classification](#03)
- [x] [4. Sentence Subjectivity and Sentiment Classification](#04)
- [x] [5. Aspect-based Sentiment Analysis](#05) 
- [x] [6. Sentiment Lexicon Generation](#06)
- [x] [7. Opinion Summarization](#07)
- [x] [8. Analysis of Comparative Opinions](#08)
- [x] [9. Opinion Search and Retrieval](#09)
- [x] [10. Opinion Spam Detection](#10)
- [x] [11. Quality of Reviews](#11)

<h2 id="01"> 1. Sentiment Analysis: A Fascinating Problem </h2>

- Introduction
    - Definition: **Sentiment analysis**, also called **opinion mining**, is the field of study that analyzes people's _opinions, sentiments, evaluations, appraisals, attitudes, and emotions_ towards entities such as _products, services, organizations, individuals, issues, events, topics, and their attributes_.
    - **Sentiment analysis** vs. **Opinion mining**  
        - Industry prefer the previous
        - Academia use both interchangeably (this book)
    - sentiments before the year 2000. Since then, the field has become a very active research area. There are several reasons:
        - a wide range of application.
        - many challenging research problems
        - we now have a huge volume of opinionated data in the social media on the Web.
- Sentiment Analysis Applications
    - I myself have implemented a sentiment analysis system called **Opinion Parser**, and worked on projects in all these areas in a start-up company. There have been **at least 40-60 start-up companies in the space in the USA alone**. 
- Sentiment Analysis Research
    - Different Levels of Analysis: 
        - Difficulty intensity is increasing: Document level, Sentence level, Entity and Aspect level
        - Regular opinions and Comparative opinions (more challenge)
    - Sentiment Lexicon and Its Issues
        - Sentiment words and phrases are instrumental to sentiment analysis for obvious reasons.
        - Sentiment lexicon is necessary but not sufficient for sentiment analysis, several issues were hightlighted:
            1. A positive or negative sentiment word may have opposite orientations in different application domains.
            2. A sentence containing sentiment words may not express any sentiment.
            3. Sarcastic sentences with or without sentiment words are hard to deal with.   
            4. Many sentences without sentiment words can also imply opinions. 
    - Natural Lanaguage Processing Issues
        - It is useful to realize that sentiment analysis is a highly restricted NLP problem because the system does not need to fully understand the semantics of each sentence or document but only needs to understand some aspects of it, i.e., positive or negative sentiments and their target entities or topics. 
- Opinion Spam Detection
    - It allows people with hidden agendas or malicious intentions to easily game the system to give people the impresssion that they are independent members of the public and post fake opinions to promote or to dicredit target products, services, organizations, or individuals without disclosing their true intentions, or the person or organization that they are secretly working for. Such individuals are called **opinion spammers** and their activities are called  **opinion samming**.

<h2 id="02"> 2. The Problem of Sentiment Analysis </h2> 

- Introduction
    - Structure problem: **It is open said that if we cannot structure a problem, we probably do not understand the problem.**
    - Unlike factual information, opinions and sentiments have an important characteristic, namely, they are subjective.
    - Big difference among different forms of opinion text: such as news articles, tweets, forum discussions, blogs, and Facebook postings. **Forum discussions are perhaps the hardest to deal with** because the users there can discuss anything and also interact with one another.
- Problem Definitions
    - Opinion Definition
        - `$(g, s)$`, a sentiment _s_ on the target _g_.
        - `$(g, s, h, t)$`, a sentiment _s_ on the target _g_ from holder _h_ in time _t_.
        - `$(e_{i}, a_{ij}, s_{ijkl}, h_{k}, t_{l})$`, a sentiment `$s_{ijkl}$` on the attribute `$a_{ij}$` of entity `$e_{i}$` from holder `$h_{k}$` in time `$t_{l}$`.
    - Sentiment Analysis Tasks
        - definition: 
            - Objective of sentiment analysis, entity category and entity expression, aspect category and aspect expression, explicit aspect expression, implicit aspect expression.
        - 6 main task:
            1. entity extraction and categorization.
            2. aspect extraction and categorization.
            3. opinion holder extraction and categorization.
            4. time extraction and standardization.
            5. aspect sentiment classification.
            6. opinion quintuple generation.
- Opinion Summarization
    - Unlike factual information, opinions are essentially subjective. One opinion from a single opinion holder is usually not sufficient for action. In most applications, one needs to **analyze opinions from a large number of people**.
- Different Types of Opinions
        - Regular opinion (direct opinion (more focus currently) vs. indirect opinion) vs. Comparative opinion (e.g. xx better than xx).
    - Explicit opinion (more focus currently) vs. Implicit opinion.
- Subjectivity and Emotion
    - Subjectivity != Sentiment, Subjectivity classification.
    - Emotions are closely related to but not equal to Sentiments. 
        - Rational evaluation vs. Emotional evaluation. To make use of these two types of evaluation in practice, we can design 5 sentiment ratings,_emotional negative(-2), rational negative(-1), neutral(0), rational positive(+1), and emotional positive(+2)_. 
- Author and Reader Standing Point
    - The author this book isn't aware of any reported studies about this issue.

<h2 id="03"> 3. Document Sentiment Classification </h2>

- Introduction
    - **Definition:** Given an opinion document _d_ evaluating an entity, determine the overall sentiment _s_ of the opinion holder about the entity, determine the overall sentiment _s_ of the opinion holder about the entity, i.e, determine _s_ expressed on aspect GENERAL in the quintuple _(\_, GENERAL, s, \_, \_)_.
    - **Assumption:** Sentiment classification or regression assumes that the opinion document _d_ (e.g., a product review) expresses opinions on a single entity _e_ and contains opinions from a single opinion holder _h_.
    - Most existing techniques for document-level classification use supervised learning, although there are also unsupervised methods. Sentiment regression has been done mainly using supervised learning. Recently, several extensions to this research have also appeared, most notably, _cross-domain sentiment classification_ or (_domain adaptation_) and _cross-language sentiment classification_, which will also be discussed at length.  
- **Sentiment Classification Using Supervised Learning**
    - Most research papers do not use the neutral class, which makes the classification problem considerably easier, but it is possible to use the neural class, e.g., assigning all 3-star reviews the neutral class.
    - The key for sentiment classification is the engineering of a set of effective features. Some of the example features are:
        - Terms and their frequency
        - Part of speech
        - Sentiment words and phrase 
        - Rules of opinions
        - Sentiment shifters
        - Syntactic dependency
- **Sentiment Classification Using Unsupervised Learning**
    - Since sentiment words are often the dominating factor for sentiment classification, it is not hard to imagine that sentiment words and phrases may be used for sentiment classification in an unsupervised manner.
    - Two approach
        1. It performs classification based on some fixed syntatic pattern that are likely to be used to express opinions. 
        2. Another unsupervised approach is the lexicon-based method, which uses a dictionary of sentiment words and phrases with their associated orientations and strength, and incorporates intensification and negation to compute a sentiment score for each document.
- **Sentiment Rating Prediction**
    - Apart from classification of positive and negative sentiments, researchers also studied the problem of predicting the rating score (e.g., 1-5 stars).
    - One-vs-All (OVA) strategy, and a meta-learning method called metric labeling. Each of the opinions is a triple, a sentiment word, a modifier, and a negator.
    - To transfer the regression model to a newly given domain-dependent application, the algorithm derives a set of statistics over the opinion scores and then use them as additional features together with the standard unigrams for rating prediction.
    - Instead of predicting the rating of each review, Snyder and Barzilay (2007) studied the problem of predicting the rating for each aspect.
- **Cross-Domain Sentiment Classification**
    - Words and even language constructs used in different domains for expressing opinions can be quite different. **Thus, domain adaptation or transfer learning is needed.** 
    - (Aue and Gamon, 2005): (1) training on a mixture of labeled reviews from other domains where such data are available and testing on the target domain; (2) training a classifier as above, but limiting the set of features to those only observed in the target domain; (3) using ensembles of classifiers from domains with available labeled data and testing on the target domain; (4) combining small amounts of labeled data with large amounts of unlabeled data in the target domain (this is the traditional semi-supervised learning setting).
    - (Tan et al., 2007) trains a base classifier to label some informative examples in the target domain, and then uses the classifier to label some informative examples in the target domain. 
    - (Blitzer, Dredze and Pereira, 2007) used a method called structural correspondence learning (SCL) for domain adaptation, which was proposed earlier in (Blitzer, McDonald and Pereira, 2006). Given labeled reviews from a source domain and unlabeled reviews from both the source and target domains.
    - (Pan et al., 2010) proposed a method similar to SCL at the high level. The algorithm works in the setting where there are only labeled examples in the source domain and unlabeled exmaples in the target domain.
    - (He, Lin and Alani, 2011) used joint topic modeling to identify opinion topics. 
    - In (Yoshida et al., 2011), the authors proposed a method for transfer from multiple source domains to multiple target domains by identifying domain dependent and independent word sentiments. 
- **Cross-Language Sentiment Classification**
    - Cross-language sentiment classification means to perform sentiment classification of opinion documents in multiple languages. 
    - Two Motivation:
        1. Researchers from different countries want to build sentiment analysis systems in their own languages. However, much of the research has been done in English. 
        2. Many applications, companies want to know and compare consumer opinions about their product and services in different countries.
    - Paper review
        - (Wan, 2008) (1) translate Chinese to English by different translator. (2) calculate sentiment by lexicon-based approach. (3) ensamble various version sentiment. 
        - (Wan, 2009) a co-training method was proposed which made use of an annotated English corpus for classification of Chinese reviews in a supervised manner. 
        - Wei and Pal (2010) proposed to use a transfer learning method of cross-language sentiment classification. 
- Summary
    - Document sentiment classification is not easily applicable to non-reviews such as forum discussions, blogs, and news articles, because many such postings can evaluate multiple entities and compare them.

<h2 id="04"> 4. Sentence Subjectivity and Sentiment Classification </h2>

- Introduction
    - Assumption: a sentence usually contains a single opinion (although not true in many cases).
    - Problem Definition: Given a sentence _x_, determine whether _x_ expresses a positive, negative, or neutral (or no) opinion.
    - The quintuple (_e, a, s, h, t_) definition is not used here because sentence-level classificaiton is an intermediate step.
    - Two step: (1) classify whether a sentence expresses an opinion or not. (due to the common practice, we still use the term _subjectivity classification_) (2) classify those opinion sentences into positive and negative classes.  
- Subjectivity Classification
    - Problem: classify sentences into two classes: subjective and objective.
    - Paper review:
        - (Wiebe, 2000) proposed an unsupervised method for subjectivity classification, which simply used the presence of subjective expressions in a sentence to determine the subjectivity of a sentence. 
        - (Yu and Hatzivassiloglou, 2003) performed subjectivity classifications using sentence similarity and a naive Bayes classifier. 
        - (Riloff and Wiebe, 2003) proposed a bootstrapping approach to label training data automatically.
        - For pattern learning, a set of syntatic templates are provided to restrict the kinds of patterns to be learned.
        - (Wiebe and Riloff, 2005) use the rule-based subjective classifier classifies a sentence as subjective if it contains two or more strong subjective clues.
        - (Riloff, Patwardhan and Wiebe, 2006) studied relationship among different features. 
        - (Pang and Lee, 2004), The algorithm works on a sentence graph of an opinion document.
        - (Raaijmakers and Kraaij, 2008), character n-grams algorithm.  
- Sentence Sentiment Classificaiton
    - If a sentence is classified as being subjective, we determine whether it expresses a positive or negative opinion.
    - Paper review:
        - (Hu and Liu, 2004), Hu and Liu proposed a lexicon-based algorithm for aspect level sentiment classification, and extent with WordNet.
        - (Gamon et al., 2005) a semi-supervised learning algorithm was used.
        - (McDonald et al., 2007), the authors presented a hierachical sequence learning model similar to conditional random fields (CRF).
        - (Hassan, Qazvinian and Radev, 2010), a method was proposed to identify attitudes about participants in online discussions. 
- Dealing with Conditional Sentences
    - Conditional sentences are sentences that describe implications or hypothetical situations and their consequences. Such a sentence typically contains two clauses: the condition clause and the consequent clause. 
- Dealing with Sarcastic Sentences
- Cross-language Subjectivity and Sentiment Classification
    1. Translate test sentences in the target language into the source language and classify them using a source language classifier.
    2. Translate a source language training corpus into the target language and build a corpus-based classifier in the target language.
    3. Translate a sentiment or subjectivity lexicon in the source language to the target language and build a lexicon-based classifier in the target language. 
- Using Discourse Information for Sentiment Classification
    - (Asher, Benamara and Mathieu, 2008) used five types of rhetorical relations: _Contrast_, _Correction_, _Support_, _Result_, and _Continuation_  with attached sentiment information for annotation. 
    - _opinion frame_, the components of opinion frames are opinions and the relationships between their targets. 
- Summary


<h2 id="05"> 5. Aspect-based Sentiment Analysis </h2>

- Introduction
    - many phrase and word sentiments depend on aspect contexts.
    - this chapter focuse on following two core tasks:
        1. Aspect extraction (This procedure can be skipped when the opinion targets were given.)
        2. Aspect sentiment classification  
- **Aspect Sentiment Classification**
    - Two approaches:
        1. supervised learning (highly rely on training data so that it has difficulty to scale up to a large number of application domains.)
        2. lexicon-based approach (This unsupervised approach can avoid some of issues and has been shown to perform quite well in a large number of domains.)
    - A typical lexicon-based algorithm (Ding, Liu and Yu, 2008), where, they assume that entities and aspects are know:
        1. Mark sentiment words and phrases.
        2. Apply sentiment shifters: Sentiment shifters are words and phrases that can change sentiment orientations.
        3. Handle but-clauses: Words or phrases that indicate _contrary_ need special handling because they often change sentiment orientations too.
        4. Aggregate opinions.     
- **Basic Rules of Opinions and Compositional Semantics**
    - An opinion rule expresses a concept that implies a positive or negative sentiment. It can be as simple as individual sentiment words with their implied sentiments or compound expressions that may need commonsense or domain knowledge to determine their orientations. This section describes some of these rules.  
- **Aspect Extraction**
    - which can also be seen as an information extraction task.
    - which can be split into implicit and explicit aspect extraction
    - here focus on 4 explicit aspect extraction strategy
        - **Finding Frequenct Nouns and Noun Phrases**: (1) POS. (2) Calculate word's weight by pointwise mutual information (PMI), frequency, tf-idf, etc.
        - **Using Opinion and Target Relations**: Since opinions have targets, they are obviously related. Their relationships can be expoited to extract aspects which are opinion targets because sentiment words are often known. (Key idea: dependency parser).
        - **Using Supervised Learning**: Aspect extraction can be seen as a special case of the general information extraction problem. The most dominant methods are based on _sequential learning_ (or _sequential labeling_), for instance HMM and CRF. There are also other approaches, like dependency tree, on-class SVM.
        - **Using Topic Models**: In recent years, statisstical topic models have emerged as a principled method for discovering topics from a large collection of text documents. And Topic models is an unsupervised method. There are two main basic models, _pLSA(Probabilisstic Latent Semantic Analysis)_ and _LDA(Latent Dirichlet allocation)_. It is note that, topics can cover both aspect words and sentiment words, for sentiment analysis, they need to be separated. Such separation can be achieved by extending the basic model (e.g. LDA) to jointly model both aspects and sentiments. 
    - **Mapping Implicit Aspects**: There are many types of implicit aspect expressions. Adjectives and adverbs are perhaps the most common types. Although explicit aspect extraction has been studied extensively, limited research has been done on mapping implicit aspects to their explicit aspects. 
- **Identifying Resource Usage Aspect**
    - In the context of aspect extraction and aspect sentiment classification, it is not always the sentiment word and aspect word pairs that are important. The real world is much more complex and diverse thant that. Here, we use resource usage as an example to show that a divide and conquer approach may be needed for aspect-based sentiment analysis. 
    - In many applications, resource usage is an important aspect, e.g., _"This washer uses a lot of water."_ 
- **Simutaneous Opinion Lexicon Expansion and Aspect Extraction**
    - An opinion always has a target. This property has been exploited in aspect extraction by several researchers, it was used to extract both sentiment words and aspects at the same time by expoiting certain syntatic relations between sentiments and targets. 
    - As the process involves propagation through both sentiment words and aspects, the method is called _double propagation_.
    - The propagation performs for subtasks:
        1. extracting aspects using sentiment words.
        2. extracting aspects using extracted aspects.
        3. extracting sentiment words using extracted aspects.
        4. extracting sentiment words using both given and extracted opinion words. 
- **Grouping Aspects into Categories**
    - After aspect extraction, aspect expressions (actual words and phrases indicating aspects) need to be grouped into synonymous aspect categories.
    - Each category represents a unique aspect. As in any writing, people often use different words and phrases to describe the same aspect.  
- **Entity, Opinion Holder and Time Extraction**
    - Entity, opinion holder and time extraction is the classic problem of named entity recognition (NER). 
    - There are two main approaches to information extraction: rule-based and statistical. 
- **Coreferennce Resolution and Word Sense Disambiguation**
    - If refers to the problem of determining multiple expressions in a sentence or document referring to the same thing. 
- Summary 
    - Two most outstanding problems are aspect extraction and aspect sentiment classifications. 
    - On the whole, we seem to have met a long tail problem. While sentiment words can handle about 60% of the cases (more in some domains and less in others), the rest are highly diverse, numerous and infrequent, which make it hard for statistical learning algorithms to learn patterns because there are simply not enough training data for them.
    - So far, the research community has mainly focused on opinions about electronics products, hotels, and restaurants.  

<h2 id="06"> 6. Sentiment Lexicon Generation </h2>

- Introduction
    - It should be quite clear that words and phrases that convey positive or negative sentiments are instrumental for sentiment analysis.
    - _sentiment words_ are also called _opinion words_, _polar words_, _opinion-bearing words_. 
    - sentiment words mean both individual words and phrases.
    - sentiment words can be divided into two types, _base type_ and _comparative type_, this chapter only focus on the previous type.
    - Three main approaches are: _manual approach, dictionary-based approach, and corpus-based approach_. 
- **Dictionary**
    - a simple technique: (1) construct a small set of sentiment words (seed) manually. (2) grows this set by searching in the WordNet or another online dictionary for their synonyms and antonyms. (3) add newly found words into the seed list. (4) back to (1) and start next iteration.
    - Paper review:
        - (Kamps et al., 2004) used a WordNet distance based method to determine the sentiment orientation of a given adjective.
        - (Blair-Goldensohn et al., 2008), a different bootstrapping method was proposed, which used a positive seed set, a negative seed set, and also a neutral seed set. 
        - (Rao and Ravichandran, 2009), three graph-based semi-supervised learning methods were tried to separate positive and negative words given a positive seed set. 
        - (Hassan and Radev, 2010) presented a Markov random walk model over a word relatedness graph to produce a sentiment estimate for a given word.
        - (Turney and Littman, 2003) PMI based method.
        - (Esuli and Sebastiani, 2005) supervised learning.
        - (Andreevskaia and Bergler, 2006) ensemble.
        - (Kaji and Kitsuregawa, 2006; Kaji and Kisuregawa, 2007), heuristics + HTML documents.
        - (Dragut et al., 2010), WordNet + deductive process.
    - advantage:
        - easily and quickly find a large number of sentiment words with their orientations. 
    - disadvantage:
        - it is hard to use the dictionary-based approach to find domain or context dependent orientations of sentiment words.    
- **Corpus-based Approach**
    - two main scenarios:
        1. given a seed list of known (often general-purpose) sentiment words, discover other sentiment words and their orientations from a domain corpus.
        2. adapt a general-purpose sentiment lexicon to a new one using a domain corpus for sentiment analysis applications in the domain.
    - papers review:
        - (Hazivassiloglou and McKeown, 1997) used a corpus and some seed adjective sentiment words to find additional sentiment adjectives in the corpus by conjunction words like AND, OR, BUT, EITHER-OR, and NEITHER-NOT. This idea is callded _sentiment consistency_.
        - Domain-dependent sentiment words and their orientations is insufficient.
        - many words in the same domain can have different orientations in different contexts.
        - expression level sentiment classificaiton.
        - The problem of adapting a general lexicon to a new one for domain specific expression level sentiment classification.
        - The problem of adapting the sentiment lexicon from one domain (not a general-purpose lexicon) to another domain.
        - The problem of producing a connotation lexicon.   
- **Desirable and Undesirable Facts**
    - Many objective words and expressions can imply opinions too in certain domains or contexts because they can represent desirable or undesirable facts in these domains or contexts. 
    - A simple approach, accuracy is still not high
        1. Condidate identification.
        2. Prunning.  
- **Summary**
    - Some public lexicons: 
        - [General Inquirer lexicon, Stone, 1968](http://www.wjh.harvard.edu/~inquirer/spreadsheet_guide.htm) 
        -  ....

<h2 id="07"> 7. Opinion Summarization </h2>

- Introduction
    - Motivation: In most sentiment analysis applications, one needs to study opinions from many people because due to the subjective nature of opinions, looking at only the opinion from a single person is usually insufficient. Some form of summary is needed. 
    - In general, opinion summarization can be seen as a form of _multi-document text summarization_. However, an opinion summary is quite different from a traditional single document or multi-document summary as an opinion summary is often centered on entities and aspects and sentiments about them, and also has a quantitative side, which are the essence of aspect-based opinion summary. 
- **Aspect-based Opinion Summarization**
    - Two main characteristics:
        1. it captures the essence of opinions: opinion targets (entities and their aspects) and sentiments about them.
        2. it is quantitative, which means that it gives the number or percent of people who hold positive or negative opinions about the entities and aspects. 
    - Some Summary visualization cases:
        -  A summary table for a entity which cover various aspects.
        -  Positive vs. Negative bar chart in various aspects for a specific entity.
        -  Comparative Positive vs. Negative bar char in varous aspects for multiple entity. 
- **Improvements to Aspect-based Opinion Summarization**
    -  Ontology tree
    -  (Tata and Di Eugenio, 2010) selects a representative for the group firstly.
    -  (Lu et al., 2010) selects aspects that capture major opinions firstly.
    -  (Ku, Liang, and Chen, 2006) performed blog opinion summarization, and produced two types of summaries: brief and detailed summaries.
    -  (Lerman, Blair-Goldensohn and McDonald, 2009) defined opinion summarization in a slightly different way and proposed three different models to perform summarization of reviews of a product. (1) _sentiment match (SM)_, (2) _sentiment match + aspect coverage (SMAC)_, (3) _sentiment-aspect match (SAM)_.
- **Contrastive View Summarization**
    - (Kim and Zhai, 2009) performed contrastive opinion summarization by extracting a set of _k_ contrastive sentence pairs from the sets. 
    - (Paul, Zhai and Girju, 2010), Their algorithm generates a macro multi-view summary and a micro multi-view summary. 
    - (Lerman and McDonald, 2009) wanted to produce contrastive summaries of opinions about two different products to highlight the differences of opinions about them. 
- **Traditional Summarization**
    - Weakness: (1) They only have limited or no consideration of target entities and aspects, and sentiments about them. Tus, they may select sentences which are not related to sentiments or any aspects. (2) There is no quantitative perspective.                                                                            

<h2 id="08"> 8. Analysis of Comparative Opinions </h2>

- Introduction
    - Definition: One can express opinions by comparing similar entities. Such opinions are called _comparative opinions_. 
    - Note: Comparative opinions are related to but are also different from regular opinions. They not only have different semantic meanings but also have different syntatic forms. 
    - There are in fact two main types of opinions that are based on comparisons: _comparative opinions_ and _superlative opinions_.
- **Problem Definitions**
    - A comparative sentence expresses a relation based on similarities or differences of more than one entity. There are several types of comparisons. They can be grouped into two main categories: _gradable comparison_ and _non-gradable comparison_.
    - **Gradable comparison:** (1) Non-equal gradable comparison, (2) Equative comparison, (3) Superlative comparison. 
    - **Non-gradable comparison:** (1) Entity A is similar to or different from entity B based on some of their shared aspects. (2) Entity A has aspect a1, and entity B has aspect a2 (a1 and a2 are usually substitutable). (3) Entity A has aspect a, but entity B does not have.
    - This chapter only focus on **gradable comparisons**. 
- **Identify Comparative Sentences**
    - In (Jindal and Liu, 2006a), it was shown that almost every comparative sentence has a keyword (a word or phrase) indicating comparison. Using a set of keywords, 98% of comparative sentences (recall = 98%) were identified with a precision of 32% based on their data set. The keywords are: (1) Comparative adjectives (JJR) and comparative adverbs (RBS). (2) Superlative adjectives (JJS) and superlative adverbs (RBS). (3) Other non-standard indicative words and phrases such as _favor_, _beat_...
    - (Jindal and Liu, 2006a) observed that comparative sentences have strong patterns involving comparative keywords.
    - Classifying comparative sentences into four types: non-equal gradable, equative, superlative, and non-gradable.
    - (Li et al. 2010) starts with a user-given pattern. 
- **Identifying Preferred Entities**
    - Unlike regular opinions, it does not make much sense to perform sentiment classification to a comparative opinion sentence as a whole because such a sentence does not express a direct positive or negative opinion. Instead, it compares multiple entites by ranking the entites based on their shared aspects to give a _comparative opinion_.
    - Comparative opinion words were divided into two categories.
        1. General-purpose comparative sentiment words, like _better, worse, etc._ 
        2. Context-dependent comparative sentiment words: like _higher, lower, etc._
- **Summary**
    - Situation: Although there have been some existing works, comparative sentences have not been studied as extensively as many other topics of sentiment analysis. Further research is still needed. 
    - One of the difficult problem is how to identify many types of non-standard or implicit comparative sentences.  

<h2 id="09"> 9. Opinion Search and Retrieval </h2>

- Introduce
    - Two typical kinds of opinion search queries:
        1. Find public opinions about a particular entity or an aspect of the entity.
        2. Find opinions of a person or organization (i.e., opinion holder) about a particular entity or an aspect of the entity (or topic).  
- **Web Search vs. Opinion Search**
    - Traditional Web Search:
        1. retrieve relevant documents/sentencess to the user query and
        2. rank the retrieved documents or sentences.
    - Opinion search needs to perform two sub-tasks:
        1. Find documents or sentences that are relevant to the query.
        2. Determine whether the documents or sentences express opinions on the query topic (entity and/or aspect) and whether the opinions are positive or negative. 
    - Ranking:
        - Traditional web: based on authority and relevance scores. 
        - Opinion Search: 
            - two objectives need to be concerned: (1) it needs to rank those opinionated documents or sentences with high utilities or information contents at the top. (2) it needs to reflect the natural distribution of positive and negative opinions.
            - One simple solution for this is to produce two rankings, one for positive opinions and one for negative opinions, and also to display the numbers of positive and negative opinions.      
- **Existing Opinionn Retrieval Techniques**
    - Current research in opinion retrieval typically treats the task as a two-stage process: 
        1. Documents are ranked by topical relevance only.
        2. Candidate relevant documents are re-ranked by their opinion scores.
    - Example system (Zhang and Yu, 2007)
        - Retrieval component: This component performs the traditional information retrieval (IR) task. It considers both keywords and concepts.
        - Opinion classification component: (1) classifying each document into one of the two categories, opinionated and not-opinionated, and (2) classifying each opinionated document as expressing a positive, negative, or mixed opinion.   
- Summary

<h2 id="10"> 10. Opinion Spam Detection </h2>

- Introduce
    - Definition: Positive opinionss often mean profits and fames for businessses and individuals, which, unfortunately, give strong incentives for people to game the system by posting _fake opinions_ or _reviews_ to promote or to discredit some target products, services, organizations, individuals, and even ideas without disclosing their true intensions, or the persson or organization that they are secretly working for. Such individualsss are called _opinion spammers_ and their activitiess are called _opinion spamming_.
    - Challenge: The key challenge of opinion spam detection is that unlike other forms of spam, it is very hard, if not impossible, to recognize fake opinions by manually reading them, which makes it difficult to find opinion spam data to help design and evaluate detection algorithms. For other forms of spam, one can recognize them fairly easily.
    - Situation: This chapter uses consumer reviewss as an example to study the problem. Little research has been done in the context of other forms of social media. 
- **Types of Spam and Spamming**
    - Three types of spam reviewss were identified in (Jindal and Liu, 2008) 
        1. fake reviews
        2. reviews about brands only
        3. non-reviews 
    - It has been shown in (Jindal and Liu, 2008) that types 2 and 3 spam reviews are rare and relatively easy to detect using supervised learning. This chapter thus focuses on type 1, fake reviews.
        1. Harmful Fake Reviews: Some of the existing detection algorithms are already using this idea by employing different types of rating deviation features.
        2. Individual and Group Spamming: In general, a spammer may work individually, or knowingly or unknowingly work as a member of a group.
        3. Types of Data, Features and Detection
            - Three main types of data have been used for review spam detection:
                - _Review content_
                - _Meta-data about the review_
                - _Product information_
            - These types of data have been used to produce many spam features. One can also classify the data into _public data_ and _site private data_.
            - Opinion Spam Detection: The ultimate goal of opinion spam detection in the review context is to identify every fake review, fake reviewer, and fake reviewer group.  
- **Supervised Spam Detection**
     - Task: two class, _fake_ and _non-fake_, classification problem.
     - Difficulty: there is no reliable fake review and non-fake review data available to train a machine learning algorithm to recognize fake reviews. 
     - Three supervised learning methods:
         - Due to the fact that there is no labeled training data for learning, Jindal and Liu (2008) exploited duplicate reviews. And cover three sets of features, _Review centric features_, _Reviewer centric features_, _Product centric features_. Finally, Logistic regression was used for model building, and  some rules can construct from results.
         - In (Li et al., 2011) attempted to identify fake reviews. In their case, a manually labeled fake review corpus was built from Epinions reviews. 
         - In (Ott et al., 2011) used Amazon Mechanical Turk to crowdsource fake hotel reviews of 20 hotels.   
- **Unsupervised Spam Detection**
    - Spam Detection based on Atypical Behaviors
        - The first technique is from (Lim et al., 2010), which identified several unusual reviewer behavior models based on different review patternss that suggest spamming. This method focusses on finding spammer or fake reviewers rather than fake reviews. The spamming behavior models are: (a) Targeting products, (b) Targeting groups, (c) General rating deviation, (d) Early rating deviation. 
        - (Jindal, Liu and Lim, 2010) formulated the problem as a data mining task of discovering unexpected class association rules. **Class association rules** are a special type of association rules (Liu, Hsu and Ma, 1998) with a fixed class attribute. 
            - Advantage: all the unexpectedness measure are defined on CARs rules, and are thus domain independent. 
            - Weakness: some atypical behaviors cannot be detected, e.g., time-related behaviors. 
    - Spam Detection Using Review Graph
        - In (Wang et al., 2011), a graph-based method was proposed for detecting spam in store or merchant reviews. This paper used a heterogeneous review graph with three types of nodes, i.e., reviewers, reviews, and stores to capture their relationships and to model spamming clues.       
- **Group Spam Detection**
    - An initial group spam detection algorithm was proposed in (Mukherjee et al., 2011), which was improved in (Mukherjee, Liu and Glance, 2012). It works in two steps: (1) Frequent pattern minning, (2) Rank groups based on a set of group spam indicators.  
- Summary
    - Although current research on opinion spam detection is still in its early stage, several effective algorithms have already been proposed and used in practice.  


<h2 id="11"> 11. Quality of Reviews </h2>

- Introduce:
    - The topic is related to opinion spam detection, but is also different because low quality reviews may not be spam or fake reviews, and fake reviews may not be perceived as low quality reviews by readers because as we discussed in the last chapter.
    - The objective of this task is to determine the quality, helpfulness, usefulness, or utility of each review. 
- **Quality as Regression Problem**
    - In this area of research, the ground truth data used for both training and testing are usually the user-helfulnesss feedback given to each review. 
- **Other Methods**
    - (O'Mahony and Smyth, 2009), a classification approach was proposed to classify helpful and non-helpful reviews. 
    - (Liu et al., 2007) divided reviews into 4 categories: "best review", "good review", "fair review", and "bad review". Manual labeling was carried out to produce the gold-standard training and testing data.
    - (Tsur and Rappoport, 2009) studied the helpfulness of book reviews. (1) identifies a set of important terms in the reviews. These terms together form a vector represeenting a virtual optimal or core review. (2) each actual review is mapped or converted to this vector representation. (3) each review is assigned a rank score based on the distance of the review to the virtual review.
    - (Moghaddam, Jamali and Ester, 2012), a new problem of personalized review quality prediction for recommendation of helpful reviews was proposed.    