# precision-recall-intervals

A Python script to compute precision/recall confidence intervals for a binary classifier.

The methodology is based on Section 3.5 of the paper
['Approximate Recall Confidence Intervals' by W. Weber (2012)](https://arxiv.org/abs/1202.2880).
The main idea is to assign the negative predictive value and the positive predictive value (i.e., precision) [Jeffreys priors](https://en.wikipedia.org/wiki/Jeffreys_prior), and then evaluate the posteriors by counting the number of false negatives in a sample from the total population of presumed negatives, and the number of false positives in a sample from the total population of presumed positives, respectively.

The inputs to the script are the following:

+ total number of candidate negatives;
+ total number of sampled candidate negatives;
+ number of false negatives;
+ number of candidate positives;
+ number of sampled candidate positives;
+ number of false positives;
+ confidence level in (0, 1).

The script returns a precision interval [p1, p2] and a recall interval [r1, r2] at the specified confidence level.

## Example

Consider a population of 41643552 features. Our hypothetical classifier classifies 39615617 features as 0 (negative) and 2027935 features as 1 (positive). We sample 7853 features from class 0 and identify 131 of those as false negatives, and 596 features from class 1 and identify 55 of those as false positives. The 0.95 confidence intervals for precision and recall are obtained as follows:

```bash
python compute.py 39615617 7853 131 2027935 596 55 0.95
Precision 0.95 confidence interval: [0.882505817635, 0.928983677706]
Recall 0.95 confidence interval: [0.701872052529, 0.768669678102]
```

## Install

Just clone this repository:

```bash
git clone https://github.com/PlatformStories/precision-recall-intervals
```

The only dependencies are numpy and scipy.
