# Computes precision/recall confidence intervals for a binary classification problem.
# Returns precision interval [p1,p2] and recall interval [r1,r2].
# Arguments:
# number of candidate negatives N0
# number of sampled candidate negatives n0
# number of false negatives fn
# number of candidate positives N1
# number of sampled candidate positives n1
# number of false positives fp
# confidence level 1-a
# The meaning of confidence level: we know that precision and recall lie in [p1,p2]
# and [r1,r2], respectively, with confidence 1-a.
# The methodology is based on Section 3.5 of the paper "Approximate Recall Confidence Intervals"
# by Webber, 2012.

import sys
import numpy as np
from scipy.stats import beta as bs
from numpy.random import beta as br

try:
	N0 = float(sys.argv[1])
	n0 = float(sys.argv[2])
	fn = float(sys.argv[3])
	N1 = float(sys.argv[4])
	n1 = float(sys.argv[5])
	fp = float(sys.argv[6])
	cl = float(sys.argv[7])
except IOError:
	print 'Not enough arguments!'

a = 1 - cl

# derive precision confidence interval based on beta posterior
p1 = bs.ppf(a/2.0, 0.5 + n1 - fp , 0.5 + fp)
p2 = bs.ppf(1 - a/2.0, 0.5 + n1 - fp , 0.5 + fp)

# number of Monte Carlo iterations to estimate recall distribution
no_iterations = 40000
recalls_monte = []
for i in range(no_iterations):
	# generate precision value based on beta posterior
	p_monte = br(0.5 + n1 - fp, 0.5 + fp)
	# generate negative predictive value based on beta posterior
	npv_monte = br(0.5 + n0 - fn, 0.5 + fn)

	# compute recall
	recalls_monte.append((N1*p_monte) / (N1*p_monte + N0*(1-npv_monte)))

# compute recall confidence interval
X = np.sort(recalls_monte)
F = np.arange(no_iterations).astype(float)/no_iterations
ind1, ind2 = np.argmin(np.abs(F - a/2.0)), np.argmin(np.abs(F - 1 + a/2.0))
r1, r2 = X[ind1], X[ind2]

print 'Precision {} confidence interval: [{}, {}]'.format(cl, p1, p2)
print 'Recall {} confidence interval: [{}, {}]'.format(cl, r1, r2)
