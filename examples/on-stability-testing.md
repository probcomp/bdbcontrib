Alexey Radul
October 2, 2015

One may be interested in the stability of results from a probabilistic
program.

In the case of BayesDB, a typical program consists of analyzing some
data table under some population model for some amount of computation,
and then running some query, and possibly inspecting some consequence
of the result.  We are thus interested in stability under
- changing the seed entropy of this process
- changing the number of models constructed
- changing the amount of analysis performed 

We may in general be interested in the stability of arbitrarily
complex objects, but the easy thing to start with is the independent
stability of a collection of boolean or 1-D numerical probes.

The baseline paradigm is therefore this:
- For each desired seed entropy
  - For each desired number of models to test
    - For each desired number of iterations of analysis
      - Perform that analysis
      - For each probe
        - Evaluate it on the resulting bdb
- And aggregate the results.

We can save work off the above in two ways, without losing any
generality of probes:
- It suffices to do just the maximum desired amount of analysis, and
  save checkpoints.
- It suffices to analyze just the maximum desired number of models,
  and evaluate the probes on subselections to assess stability with
  fewer models.

We can also save some complexity (but not computation) by noting that
analyzing k*n models with one seed is equivalent to analyzing n models
k times with different seeds (because the chains are actually
independent -- resampling, component swapping, or any such nonsense
would invalidate this observation).

Architecture
============

We break stability assessment into four conceptual phases, separated
by on-disk files:
- Analysis, which produces a collection of .bdb files with varying
  amounts of analysis done (on the same number of models, over the
  same dataset);
- Probing, which consumes such a collection of .bdb files and
  produces a file of serialized probe results;
- Visualization, which consumes such a file of probe results and
  draws plots that give visual indications of stability; and
- [Todo] Assessment, which consumes a file of probe results and
  produces a decision: "stable enough" or "not stable enough".

Why break into stages?  Because doing so gives room to memoize and
distribute the analysis and/or probe phases, for
- application of more compute
- iteration on downstream code

Why separate visualization and assessment?  Because assessment is what
you want for an automated build, and visualization is what you want
for developing the assessment's decision rules and for debugging build
failures.

I considered the alternative of fusing the analysis and probing phases
into one program.  My considerations and rationale for choosing to
separate are:
- Pro: can iterate on probe set without redoing analysis
- Pro: can distribute analysis separately from probing
- Con: .bdbs are actually quite big: 12MB per 60-model-satellites file
  makes 360MB per 30-point analysis series makes 36GB for 100 per-seed
  replications of a 30-point series.
  - Offset: sqlite's vacuum command helps, around 4x
  - Offset: zlib seems to compress vacuumed bdbs around 5x
- Con: The probes may take a long time to run too (either b/c of
  loading the files or b/c of compute cost per probe)
  - Offset: can still parallelize and distribute probing if needed
- Pro: The analysis-checkpointing program can be refactored relatively
  easily to fuse probing into it and avoid saving all the .bdbs.

Implementation on Satellites
============================

The analysis phase is carried out by the program
`examples/satellites/build_bdbs.py`
- This constructs a directory with a bunch of bdb files.
- The files are named according to a predictable pattern, and
- Metadata about them is saved in the directory as well.
- The program is configured by editing global variables in its
  source code.

The probing phase is carried out by the program
`examples/satellites/probe.py`
- The probe set and configuration is defined in the source of the
  program.

The visualization phase is carried out by the program
`examples/satellites/visualize.py`
- The configuration is defined in the source of the program.
- Visualization content is determined entirely by the probe results.

Generic probing and visualization machinery is available in the
modules `bdbcontrib.experiments.probe.py` and
`bdbcontrib.experiments.visualization.py`.

Possible Extensions
===================

The behavior of probes that are directly computed as Monte-Carlo
estimates over the models can be predicted using the Central Limit
Theorem.

To wit, suppose some probe F : bdb -> R is actually computed as the
average over all models in the bdb of an underlying f : model -> R.
Then, if f has finite mean and variance, the CLT predicts that as the
number of models increases, the distribution on F will become a
Gaussian, whose mean and variance can be predicted from the mean and
variance of f.  Further, if f has finite skew, the Berry-Esseen
theorem predicts the maximum Kolmogorov-Smirnoff distance of F from
the predicted Gaussian, in terms of the skew of f.

The mean, variance, and skew of f can be estimated from (the
computation performed during) a single evaluation of F on all
available models.  If the assumptions are believed, the stability of F
for any number of models can then be predicted without requiring
replications.
- In effect, we save n_replications work.
- If we are willing to extrapolate for more models than the original
  experiment, we save an unbounded amount of work.
- Assumption: The mean, variance, and skew of the sample are good
  estimates of the true mean, variance, and skew of f.

This could be implemented directly by adding a 'monte-carlo' result
type to the aggregation framework, which would maintain all the
desired statistics, and performing the above prediction in the
visualization or assessment phases.

This can also in principle be extended to predicting the behavior of
probes that evaluate a function G of several monte-carlo quantities
F_i.  If we either assume the underlying f_i are independent, or if we
use a multivariate version of the CLT, we can propagate the predicted
distributions on F_i through the computation of G to predict a
distribution on G.
- Again saving n_replications (or more) work.
- Doing this in general, however, requires a little probabilistic
  programming language for propagating Gaussian distributions through
  probe computations.
- Option: Implement said language by forward sampling.  That is, use
  the Gaussian approximation to generate many more inputs to the probe
  than we can afford with the underlying computation
  - Plus: Can study the probe arbitrarily finely by sampling repeatedly
  - Plus: Can project to higher true effort by lowering the variance
    of the Gaussians (but uncertainty about true means remains)
  - Minus: An exact analysis of the probe could be faster (by saving
    probe computation iterates) (but this is irrelevant if analysis
    time dominates probing time)
  - Minus: An exact analysis may be better able to use the
    Berry-Esseen theorem to bound the effect of non-Gaussian-ness
    - But there are still things one can say, e.g. refusing to do it
      if the projected K-S stat is too high.

The same machinery can also be used in BayesDB, in at least two ways:
- Predicting for our users how variable certain query results may be,
  at no additional work.
  - This applies to all queries that have numerical answers, that is,
    everything except simulate and predict_confidence.
- Teaching downstream functions to have a notion of confidence.
  - e.g. an ORDER BY WITH CONFIDENCE variant that, when the ordering
    column is an estimated property like PREDICTIVE PROBABILITY, does
    the Gaussian mass comparison and only puts one thing above another
    if the probability of them being ordered that way is above the
    confidence.
    
Note: There may also be a version of this whole story for
likelihood-weighted monte carlo estimation.
- c.f. Pareto-stabilized importance sampling
