# Machine Learning Yearning, NG, Andrew, 2018
Finished at 20190224.

**1. Why Machine Learning Strategy**  
**2. How to use this book to help your team**  
**3. Prerequisites and Notation**  

>This book will tell you how. Most machine learning problems leave clues that tell you what’s useful to try, and what’s not useful to try. Learning to read those clues will save you months or years of development time.  
A few changes in prioritization can have a huge effect on your team’s productivity

**4. Scale drives machine learning progress** 

>Data availability and Computational scale.   
Common and more reliable way: Train a bigger network or get more data.   
If #samples <= 20 Then: traditional algorithm + feature engineering  
Elif #samples > 1 million Then: Neural Network(NN)

## Setting up development and test sets
 
**5. Your development and test set**
> Training set: Which you run your learning algorithm on.  
> Dev(development) set: Which you use to tune parameters, select features, and make other decisions regarding the learning algorithm. Sometimes also the _hold-out cross validation set_  
> Test set: which you use to evaluate the performance of the algorithm, but not to make any decisions regarding what learning algorithm or parameters to use. 
>  
> !!! Choose deve and test set to reflect data you expect to get in the future and want to do well on.

**6. Your dev and test sets should come from the same distribution**
> If not, your options are less clear. Several things could have gone wrong:  
> 1. You had overfit to the dev set.  
> 2. The test set harder than the dev set.  
> 3. The test set not necessarily harder, but just different, from the deve set.  
> 
> !!! In common case, avoiding transfer learning to make work harder. 

**7. How large do the dev/test sets need to be?**
> dev set: be large enough to _detect difference_ between algorithms   
> test set: be large enough to give _high confidence_ in the overall performance of your system.
> 
> !!! The old heuristic of a 70%/30% train/test split does not apply for problems where you have a lot of data; the dev and test sets can be much less than 30% of the data.  
> !!! There is no need to have excessively large dev/test sets beyond what is needed to evaluate the performance of your algorithms.

**8. Establish a single-number evaluation metric for your team to optimize**
> Problem: Having multiple-number evaluation metrics makes it harder to compare algorithms. Suppose
> Advantages: (1) to speeds up your ability to make a decision when you are selecting among a large number of classifiers. (2) gives a clear preference ranking among all of them, and therefore a clear direction for progress.  
> Common strategy: Taking an average or weighted average is one of the most common ways to combine multiple metrics into one.

**9. Optimizing and satisficing metrics**
> Problem: tradeoff accuracy and running time   
> Strategy: (1) define “acceptable” running time; (2) maximize accuracy.  
> General: for N different criteria, set N-1 of the criteria as “satisficing” metrics.

**10. Having a dev set and metric speeds up iterations**  
> Loop: idea --> code --> experiment --> idea ...  
> !!! This is an iterative process. The faster you can go round this loop, the faster you will make progress.  
> !!! Machine learning is a highly iterative process: You may try many dozens of ideas before finding one that you’re satisfied with.

**11. When to change dev/test sets and metrics**
> I typically ask my teams to come up with an initial dev/test set and an initial metric in _less than one week—rarely longer_.  
> If you later realize that your initial dev/test set or metric missed the mark, by all means change them quickly.
> 
> Example: if your dev set + metric ranks classifier A above classifier B, but your team thinks that classifier B is actually superior for your product, then this might be a sign that you need to change your dev/test sets or your evaluation metric.   
> There are three main possible causes of the dev set/metric incorrectly rating classifier A higher:  
> 1. The actual distribution you need to do well on is different from the dev/test sets.  
> 2. You have overfit to the dev set. (If you need to track your team’s progress, you can also evaluate your system regularly—say once per week or once per month—on the test set)  
> 3. The metric is measuring something other than what the project needs to optimize.
> 
> !!! If you ever find that the dev/test sets or metric are no longer pointing your team in the right direction, it’s not a big deal! Just change them and make sure your team knows about the new direction.

**12. Takeaways: Setting up development and test sets**

## Basic Error Analysis

**13. Build your first system quickly, then iterate**
> When you start a new project, especially if it is in an area in which you are not an expert, it is hard to correctly guess the most promising directions.
> 
> So, Don’t start off trying to design and build the perfect system. Instead, build and train a basic system quickly—perhaps in just a few days. Then use error analysis to help you identify the most promising directions and iteratively improve your algorithm from there.

**14. Error analysis: Look at dev set examples to evaluate ideas**
> Error analysis can often help you figure out how promising different directions are.  
> 
> Example: A team member proposes incorporating 3rd party software that will make the system do better on dog images. These changes will take a month, and the team member is enthusiastic. Should you ask them to go ahead?  
> Error analysis: (1) Gather a misclassified sample from dev set. (2) Look at these examples manually, and count what fraction of them are dog images.
> 
> !!! _Error Analysis_ refers to the process of examining dev set examples that your algorithm misclassified, so that you can understand the underlying causes of the errors. This can help you prioritize projects—as in this example—and inspire new directions.

**15. Evaluating multiple ideas in parallel during error analysis**
> Methodology: I create a spreadsheet and fill it out while looking through misclassified dev set, and also jot down comments that might help me remember specific examples.   
> \<error-category-1, error-category-2,....,error-category-n, comment\> ---> error analysis.
>
> !!! The most helpful error categories will be ones that you have an idea for improving. But don't limit to that, the goal of this process is to build your intuition about the most promising areas to focus on.  
> !!! Error analysis is an iterative process. Don’t worry if you start off with no categories in mind. After looking at a couple of images, you might come up with a few ideas for error categories. After manually categorizing some images, you might think of new categories and re-examine the images in light of the new categories, and so on.

**16. Cleaning up mislabeled dev and test set example**
> “mislabeled” here: mean that the pictures were already mislabeled by a human labeler even before the algorithm encountered it.
> 
> !!! It is not uncommon to start off tolerating some mislabeled dev/test set examples, only later to change your mind as your system improves so that the fraction of mislabeled examples grows relative to the total set of errors. 
 
**17. If you have a large dev set, split it into two subsets, only one of which you look at**  
> You will more rapidly overfit the portion that you are manually looking at(Eyeball dev set). You can use the portion you are not manually looking at to tune parameters(Blackbox dev set).  
> Explicitly splitting your dev set into Eyeball and Blackbox dev sets allows you to tell when your manual error analysis process is causing you to overfit the Eyeball portion of your data.

**18. How big should the Eyeball and Blackbox dev sets be?**
> Eyeball dev set should be big enough so that your algorithm misclassifies enough examples for you to analyze:   
> \#Eyeball dev set <= 10 mistakes: very small, it’s hard to accurately estimate the impact of different error categories, but better than nothing for small data set.  
> \#Eyeball dev set ~ 20 mistakes: you would start to get a _rough sense_ of the major error sources.   
> \#Eyeball dev set ~ 50 mistakes: you would get a _good sense_ of the major error sources.  
> \#Eyeball dev set ~ 100 mistakes: you would get a _very good sense_ of the major sources of errors.  
> 
> !!! I’ve seen people manually analyze even more errors—sometimes as many as 500. There is no harm in this as long as you have enough data.  
> !!! If performance on the Eyeball dev set is much better than the Blackbox dev set, you have overfit the Eyeball dev set and should consider acquiring more data for it.  
> !!! If you are working on a task that even humans cannot do well, then the exercise of examining an Eyeball dev set will not be as helpful because it is harder to figure out why the algorithm didn’t classify an example correctly. In this case, you might omit having an Eyeball dev set.  
> 
> Blackbox dev set of 1,000-10,000 examples will often give you enough data to tune hyperparameters and select among models, though there is little harm in having even more data. A Blackbox dev set of 100 would be small but still useful.
> 
> !!! If your dev set is not big enough to split this way, just use the entire dev set as an Eyeball dev set for manual error analysis, model selection, and hyperparameter tuning.      
> !!! Between the Eyeball and Blackbox dev sets, I consider the Eyeball dev set more important

**19. Takeaways: Basic error analysis**

## Bias and Variance

**20. Bias and Variance: The two big source of error**
> Motivation: Even though having more data can’t hurt, unfortunately it doesn’t always help as much as you might hope. It could be a waste of time to work on getting more data. So, how do you decide when to add data, and when not to bother?
> 
> Example: Your algorithm has 15% (or 85% accuracy) on training set while 16% error (84% accuracy) on the dev set.  
> Informal Definition:  
> algorithm’s bias:the algorithm’s error rate on the training set, where is 15%.   
> algorithm’s variance: how much worse the algorithm does on the dev (or test) set than the training set, where is 1%.  
> 
> Changes and effects:  
> bias: improve its performance on the training set.   
> variance: help it generalize better from training set to dev set.   
> 
> !!! Error = Bias + Variance. But for our purposes of deciding how to make progress on an ML problem, the more informal definition of bias and variance given here will suffice.  
> !!! There are also some methods that can simultaneously reduce bias and variance, by making major changes to the system architecture. But these tend to be harder to identify and implement.

**21. Examples of Bias and Variance**  
> high bias? high variance? both of them? ---> neither of them? (well done)
 
**22. Comparing to the optimal error rate**  
> the “ideal” error rate is nearly 0%: bias vs. variance.  
> 
> For problems where the optimal error rate is far from zero, here's a more detailed breakdown of an algorithm's error.  
> Example (speech recognition): training error is 15%, dev error is 30%, optimal error rate is 14%.    
> dev set error(30%) = Optimal error rate (“unavoidable bias”) + Avoidable bias:   
> > Optimal error rate (“unavoidable bias”) : 14% = optimal error rate.   
> > Avoidable bias : 1% = training error - optimal error rate. (If this number is negative, means you are overfitting on the training set.)  
> > Variance : 15% =  dev error - training error.
> 
> !!! The concept of variance remains the same as before. In theory, we can always reduce variance to nearly zero by training on a massive training set. Thus, all variance is “avoidable” with a sufficiently large dataset, so there is no such thing as “unavoidable variance.”  
> !!! In statistics, the optimal error rate is also called "Bayes error rate", or Bayes rate.  
> !!! How do we know what the optimal error rate is? For tasks that humans are reasonably good at, such as recognizing pictures or transcribing audio clips, you can ask a human to provide labels then measure the accuracy of the human labels relative to your training set. This would give an estimate of the optimal error rate. If you are working on a problem that even humans have a hard time solving (e.g., predicting what movie to recommend, or what ad to show to a user) it can be hard to estimate the optimal error rate.

**23. Addressing Bias and Variance**
> simplest formula for addressing bias and variance issues
> > high avoidable bias --> increase the size of your model (for example, increase the size of your neural network by adding layers/neurons).  
> > high variance --> add data to your training set.
> 
> !!! If you are using neural networks, the academic literature can be a great source of inspiration.  
> !!! If you increase the model size with regularization, usually your performance will stay the same or improve; it is unlikely to worsen significantly. The only reason to avoid using a bigger model is the increased computational cost.   

**24. Bias vs. Variance tradeoff**  
**25. Techniques for reducing avoidable bias**  
> Increase the model size (such as number of neurons/layers).   
> Modify input features based on insights from error analysis. In theory, adding more features could increase the variance; but if you find this to be the case, then use regularization, which will usually eliminate the increase in variance.   
> Reduce or eliminate regularization (L2 regularization, L1 regularization, dropout). This will reduce avoidable bias, but increase variance.  
> Modify model architecture (such as neural network architecture) so that it is more suitable for your problem: This technique can affect both bias and variance.
> 
> One method that is not helpful:  
> Add more training data : This technique helps with variance problems, but it usually has no significant effect on bias.

**26. Error analysis on the training set**  
**27. Techiniques for reducing variance**
> Add more training data : This is the simplest and most reliable way to address variance.  
> Add regularization (L2 regularization, L1 regularization, dropout): This technique reduces variance but increases bias.  
> Add early stopping (i.e., stop gradient descent early, based on dev set error): This technique reduces variance but increases bias. Early stopping behaves a lot like regularization methods, and some authors call it a regularization technique.  
> Feature selection to decrease number/type of input features: This technique might help with variance problems, but it might also increase bias. so long as you are not excluding too many useful features. In modern deep learning, when data is plentiful, there has been a shift away from feature selection, and we are now more likely to give all the features we have to the algorithm and let the algorithm sort out which ones to use based on the data. But when your training set is small, feature selection can be very useful.  
> Decrease the model size (such as number of neurons/layers): Use with caution.However, I don’t recommend this technique for addressing variance.The advantage of reducing the model size is reducing your computational cost and thus speeding up how quickly you can train models.
> 
> Here are two additional tactics, repeated from the previous chapter on addressing bias:  
> Modify input features based on insights from error analysis.  
> Modify model architecture (such as neural network architecture) so that it is more suitable for your problem.

## Learning curves

**28. Diagnosing bias and variance: Learning curves**  
> axis: training set size vs. error.  
> lines: dev error, desired performance.  
> Function: Guess how much closer you could get to the desired level of performance by adding more data. 

**29. Plotting training error**   
> Your dev set (and test set) error should decrease as the training set size grows. But your training set error usually increases as the training set size grows.
> 
> axis: training set size vs. error.  
> lines: dev error, desired performance, training error.  

**30. Interpreting learning curves: High bias**    
**31. Interpreting learning curves: Other cases**  
**32. Plotting learning curves**  
> You might find that the curve looks slightly noisy (meaning that the values are higher/lower than expected) at the smaller training set sizes.  
> Having a small training set means that the dev and training errors may randomly fluctuate.  
> heavily skewed toward one class, or huge number of classes --> the chance of selecting an especially “unrepresentative” or bad training set is also larger.  
> 
> two solutions:  
> > Instead of training just one model on 10 examples, instead select several (say 3-10) different randomly chosen training sets of 10 examples by sampling with replacement 10 from your original set of 100.Compute and plot the average training error and average dev set error. (In practice, sampling with or without replacement shouldn’t make a huge difference, but the former is common practice.)  
> > If your training set is skewed towards one class, or if it has many classes, choose a “balanced” subset instead of 10 training examples at random out of the set of 100. For
> 
> I would not bother with either of these techniques unless you have already tried plotting learning curves and concluded that the curves are too noisy to see the underlying trends.
> 
> Finally, plotting a learning curve may be computationally expensive. Thus, instead of evenly spacing out the training set sizes on a linear scale as above, you might train models with 1,000, 2,000, 4,000, 6,000, and 10,000 examples.

## Comparing to human-level performance

**33. Why we compare to human-level performance**
> 1. Ease of obtaining data from human labelers.
> 2. Error analysis can draw on human intuition.
> 3. Use human-level performance to estimate the optimal error rate and also set a “desired error rate.”
> 
> There are some tasks that even humans aren’t good at. With these applications, we run into the following problems:
> > 1. It is harder to obtain labels.
> > 2. Human intuition is harder to count on.
> > 3. It is hard to know what the optimal error rate and reasonable desired error rate is.

**34. How to define human-level performance**
> An example on medical imaging application.

**35. Surpassing human-level performance**
> More generally, so long as there are dev set examples where humans are right and your algorithm is wrong, then many of the techniques described earlier will apply. This is true even if, averaged over the entire dev/test set, your performance is already surpassing human-level performance.  
> Progress is usually slower on problems where machines already surpass human-level performance, while progress is faster when machines are still trying to catch up to humans.

## Training and testing on different distributions

**36. When you should train and test on different distributions**  
>**Choose dev and test sets to reflect data you expect to get in the future and want to do well on**
>
>!!! We will continue to assume that your dev data and your test data come from the same distribution. But it is important to understand that different training and dev/test distributions offer some special challenges.  
>!!! If you train on dataset A and test on some very different type of data B, luck could have a huge effect on how well your algorithm performs.

**37. How to decide whether to use all your data**
> But in the modern era of powerful, flexible learning algorithms—such as large neural networks—the risk of missmatch has greatly diminished. This observation relies on the fact that there is some x —> y mapping that works well for both types of data.
> 
> Adding the additional 20,000 images has the following effects:
> > It gives your neural network more examples of what cats do/do not look like.  
> > It forces the neural network to expend some of its capacity to learn about properties that are specific to internet images.If these properties differ greatly from mobile app images, it will “use up” some of the representational capacity of the neural network. Theoretically, this could hurt your algorithms’ performance.
> 
> To describe the second effect in different terms, we can turn to the fictional character **Sherlock Holmes, who says that your brain is like an attic; it only has a finite amount of space. He says that “for every addition of knowledge, you forget something that you knew before. It is of the highest importance, therefore, not to have useless facts elbowing out the useful ones.**
> 
> If you do not have a big enough neural network (or another highly flexible learning algorithm), then you should pay more attention to your training data matching your dev/test set distribution.

**38. How to decide whether to include inconsistent data**
> Suppose you want to learn to predict housing prices in New York City. Given the size of a house (input feature x), you want to predict the price (target label y).  
> Given the same size x, the price of a house y is very different depending on whether it is in New York City or in Detroit. If you only care about predicting New York City housing prices, putting the two datasets together will hurt your performance. In this case, it would be better to leave out the inconsistent Detroit data.

**39. Weighting data**
> If you don’t have huge computational resources, you could give the internet images a much lower weight as a compromise.  
> $min_{\theta}\sum_{(x,y) \in MobileImg}(h_{\theta}(x)-y)^{2} + \beta\sum_{(x,y) \in InternetImg}(h_{\theta}(x)-y)^{2}$

**40. Generalizing from the training set to the dev set**
> Wrong list
> > 1. high (avoidable) bias.
> > 2. high variance.
> > 3. data mismatch.
> 
> You now have four subsets of data for analysis:  
> > 1. Training set: This is the data that the algorithm will learn.(e.g., Internet images + Mobile images)
> > 2. Training dev set: This data is drawn from the same distribution as the training set. This is usually smaller than the training set; it only needs to be large enough to evaluate and track the progress of our learning algorithm.(e.g., Internet images + Mobile images)
> > 3. Dev set: This is drawn from the same distribution as the test set, and it reflects the distribution of data that we ultimately care about doing well on.(E.g., mobile images.)
> > 4. Test set: This is drawn from the same distribution as the dev set.(E.g., mobile images.)

**41. Identifying Bias, Variance, and Data Mismatch Errors**
> (avoidable) bias: training error - optimal error  
> variance: training error - training-dev error, or  training error - dev/test error
> data mismatch: dev/test error - training-dev error 

**42. Addressing data mismatch**
> I recommend that you: (i) Try to understand what properties of the data differ between the training and the dev set distributions. (ii) Try to find more training data that better matches the dev set examples that your algorithm has trouble with. (There is also some research on “domain adaptation”—how to train an algorithm on one distribution and have it generalize to a different distribution. These methods are typically applicable only in special types of problems and are much less widely used than the ideas described in this chapter)
> 
> Error analysis: (1) On dev set; (2) double check training set vs. training dev set.  
> 
> Unfortunately, there are no guarantees in this process. For example, if you don't have any way to get more training data that better match the dev set data, you might not have a clear path towards improving performance

**43. Artificial data synthesis**  
> Keep in mind that artificial data synthesis has its challenges: it is sometimes easier to create synthetic data that appears realistic to a person than it is to create data that appears realistic to a computer.  
> When working on data synthesis, my teams have sometimes taken weeks before we produced data with details that are close enough to the actual distribution for the synthesized data to have a significant effect. But if you are able to get the details right, you can suddenly access a far larger training set than before.


## Debugging inference algorithms

**44. The Optimization Verification test**
> $ Output = argmax_{S}Score_{A}(S) $  
> There are now two possibilities for what went wrong:
> > 1. Search algorithm problem.
> > 2. Objective (scoring function) problem.

**45. General from of Optimization Verification test**
> You can apply the Optimization Verification test when, given some input x , you know how to compute Scorex(y) that indicates how good a response y is to an input x

**46. Reinforcement learning example**  
> comparing to human-level performance.

## End-to-end deep learning

**47. The rise of end-to-end learning**  
> Sentiment classification:
> > Original text --> (Parser) --> (Sentiment Classifier) --> Output  
> > Original text ----> (Learning algorithm) ----> Outout  
>
> In problems where data is abundant, end-to-end systems have been remarkably successful. But they are not always a good choice. The next few chapters will give more examples of end-to-end systems as well as give advice on when you should and should not use them.

**48. More end-to-end learning examples**  
> Even though end-to-end learning has seen many successes, it is not always the best approach. For example, end-to-end speech recognition works well. But I’m skeptical about end-to-end learning for autonomous driving. The

**49. Pros and cons of end-to-end learning**  
> These hand-engineered components limit the potential performance of the speech system. However, allowing hand-engineered components also has some advantages.  
> Having more hand-engineered components generally allows a speech system to learn with less data.  
> End-to-end learning systems tend to do well when there is a lot of labeled data for “both ends”—the input end and the output end. In this example, we require a large dataset of (audio, transcript) pairs. When this type of data is not available, approach end-to-end learning with great caution.  

**50. Choosing pipeline components: Data availability**  
> One important factor is whether you can easily collect data to train each of the components.

**51. Choosing pipeline components: Task simplicity**  
> How simple are the tasks solved by the individual components?
> 
> Machine learning does not yet have a good formal definition of what makes a task easy or hard. With the rise of deep learning and multi-layered neural networks, we sometimes say a task is “easy” if it can be carried out with fewer computation steps (corresponding to a shallow neural network), and “hard” if it requires more computation steps (requiring a deeper neural network). But these are informal definitions.
> 
> In summary, when deciding what should be the components of a pipeline, try to build a pipeline where each component is a relatively “simple” function that can therefore be learned from only a modest amount of data.

**52. Directly learning rich outputs**.   
> One of the most exciting developments in end-to-end deep learning is that it is letting us directly learn y that are much more complex than a number. In the image-captioning example above, you can have a neural network input an image(x) and directly output a caption(y).

## Error analysis by parts

**53. Error analysis by parts**  
> Our description of how you attribute error to one part of the pipeline has been informal so far: you look at the output of each of the parts and see if you can decide which one made a mistake. This informal method could be all you need.

**54. Attributing error to one part** 
> Example: Image --> (Cat dector) --> (Cat breed classifier) --> Output.   
> Case 1: Even given a “perfect” bounding box, the cat breed classifier still incorrectly outputs y=0. In this case, clearly the cat breed classifier is at fault.  
> Case 2: Given a “perfect” bounding box, the breed classifier now correctly outputs y=1. This shows that if only the cat detector had given a more perfect bounding box, then the overall system’s output would have been correct. Thus, attribute the error to the cat detector.  
 
**55. General case of error attribution**  
> The components of an ML pipeline should be ordered according to a Directed Acyclic Graph (DAG)

**56. Error analysis by parts and comparison to human-level performance**  
> There is no one “right” way to analyze a dataset, and there are many possible useful insights one could draw. Similarly, there is no one “right” way to carry out error analysis. Through these chapters you have learned many of the most common design patterns for drawing useful insights about your ML system, but you should feel free to experiment with other ways of analyzing errors as well.
> 
> This is another advantage of working on problems that humans can solve--you have more powerful error analysis tools, and thus you can prioritize your team’s work more efficiently.

**57. Spotting a flawed ML pipeline**   
> What if each individual component of your ML pipeline is performing at human-level performance or near-human-level performance, but the overall pipeline falls far short of human-level? This usually means that the pipeline is flawed and needs to be redesigned.

## Conclusion

**58. Building a superhero team - Get your teammates to read this.**