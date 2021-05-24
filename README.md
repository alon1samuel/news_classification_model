# news_classification_model
domain-specific classification of documents in: medicine, space, cryptography and electronics

Interview Problem
=================

We have customers who are interested in domain-specific analyses of documents in these fields:

- medicine
- space
- cryptography
- electronics

We intend to tag our document stream (of news articles, blogs, social media postings) with these topics so that we can identify subsets of articles for more intensive domain-specific processing.

Please implement a topic classifier and compare it to a baseline such as a logistic regression classifier. Which, if either, is better? Which should we productionize?

We are looking for this to be done as a Python project. Please email it to us in a zip archive. It could include Jupyter notebooks or documented Python scripts, it's up to you, but please make sure that it includes instructions and/or automation so that we can run it "out-of-the-box" here.

Your model does not have to be novel or achieve SOTA results. For example, it's fine to take and train an existing model. We're more interested in seeing your approach to:

- addressing the business problem
- evaluating the classifiers
- optimizing your classifier
- ensuring the research is reproducible and fit for sharing
- thinking about taking this further - scaling and generalizing

Some advice:

- the 20 News Groups dataset conveniently covers the topics above and more
- that dataset is quite small (10k documents over 20 topics) so choose your model appropriately
- `scikit-learn` has both this dataset and a logistic regression classifier
