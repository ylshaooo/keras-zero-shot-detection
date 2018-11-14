# Analysis

These scripts are used for data analysis during experiments, especially going deep into
relations among class semantics.  

Change the configurations as you perform the experiments.

---

1. If you want to know how well the model localize the objects and what the classification
results (e.g. number of the true classified and false classified) of the bounding boxes
are, run `result_matrix.py`.

2. If you want to know the similarity of a certain category in the whole dataset, run 
`similarity.py`. It is essential during objectness proposal (prediction of
confidence) period. Similarity is calculated by the cosine distance between the word
embedding of the certain classes.

3. If you want to know the similarity relationship among similar classes (e.g. dogs, cats,
and cows) so that to better infer the objectness when testing unseen class images, you can
get the confusion matrix by running `similar_matrix.py`.
