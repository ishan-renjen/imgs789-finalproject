# imgs-789-finalproject
IMGS-789 Final Project - Open-World Object Detection(OWOD) adaptation of classifying unknowns as a class/superclass category with a few-shot learning model.

**disclaimer**: my laptop died, as in it made a strange sound and shut down and wont turn on, so this code probably is not up to date. Once i get back into the CE cluster, I can push my changes but if you see this it probably isnt up-to-date but will be asap. I did try a new laptop but git is not working and neither is the RIT VPN. 

Goal: be able to assign labels to unknowns based on their embeddding with the hyperbolic or euclidean space created  by the model (whatever ends up being easier to work with)

The model uses a DETR to classify objects, with additions of converting the embeddings into a hyperbolic space and regularizing that into superclasses, based on the features in the embedding (Superclass Regularizer in the paper). Based on this, a natural extension would be if the embeddings are being regularized in some way, or used to predict unknown objects based on the current setting of the image, there should be some way to classify what catgeory at least the unknowns belong to. Giving them some form of classification would make the unknown detections usable. This method, or classifying unknown objects as a "zero-shot" problem (not really zero-shot but can be treated as such) has been attempted with open-set problems using foundation models. However, foundation models come with their own set of difficulties. 

So far, I have gone through the dataset and made sure i understood how the model works so I can adapt it. After finishing reading the papers and settling on something that (i think) is doable, I have worked on analyzing the embedding space. I collected embeddings and labels(known labels and unknowns), matched by the model, and generated some graphs. These can be seen in analzye_hyperbolic_embeddings.ipynb. Since the embeddings are high-dimensional (1x256), I used PCA down to 50 to analyze fewer dimensions which have higher variance to attempt to see where the clustering is happening. I plotted it with a t-SNE plot. This was done for the evaluation task 1-4 data with the euclidean and hyperbolic embeddings. The hyperbolic graphs are not very useful since it is trying to force euclidean geometry in a non-euclidean space but still maybe?? serves some purpose. The euclidean graphs do show consistent clustering. The location of the clusters drifts, but how much of that is due to the t-SNE plot, I am not 100% sure. I am working on developing prototype embeddings per class using K-Means, computing the euclidean distance between them (not really an ideal metric i know, but euclidean is easily available through numpy and can be an initial metric) by averaging the drift of the singular prototypes in the set per class matched with hungarian matching, and analyzing the drift to see if a prototypical network would work. This code is in analyze_prototypes.ipynb. Warning: my code is a mess and kind of unreadable since it is just for understanding. 

I am still working on getting the cost matrix defined with a hyperbolic distance metric to analyze the hyperbolic embeddings the same way. Once i get the drifts organized in a matrix for each class, then I can understand what's going on and the rest of the project should be pretty straightforward - understanding the model and code and the embedding space is always the hardest part. I did notice in the ablations in the paper, that the hyperbolic space and superclass regularizer do contribute some benefit to the results, but it still performs close enough with the euclidean space. Analyzing both would give me a better understanding of how to predict superclass categories, and maybe even get to the class categories. Especially since the euclidean space is far easier to analyze. If euclidean distance looks bad, I can change it out for mayyybe cosine similarity?? or use a covariance matrix and get the Mahalanobis distance with scipy.

My plan for the model is to create a "class-incremental" prototypical network which can continually adapt to the changing superclass embeddings. This is something I am actively looking into how to do, and will probably have a working solution by Monday and will be able to get to work on that. If the prototype drift in the euclidean space looks better than hyperbolic, i'll use the euclidean with a covariance-based distance metric since euclidean is probably not ideal. otherwise i will use the hyperbolic and learn the hyperbolic distance for a metric learning algorithm. If the prototype drift looks pretty big, which based on the paper i dont think it would, then I will have to rethink my plan and not use a prototypical network.

I have not made as much progress on this as I would have liked, but it has been an incredibly busy semester, and for the next month I don't have nearly as much going on as I do now so I can focus on this a lot more.

Papers read - mainly focused on OWOD, but read some papers on hybrid models with open-set vocabulary and one on prototypical networks for few-shot learning.
1. https://arxiv.org/pdf/2103.02603 - https://github.com/JosephKJ/OWOD
2. https://arxiv.org/pdf/2003.08798 - https://github.com/JosephKJ/iOD
3. https://openaccess.thecvf.com/content/CVPR2022/papers/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.pdf - https://github.com/akshitac8/OW-DETR
4. https://arxiv.org/pdf/2203.03800 - https://github.com/deeplearning-wisc/stud
5. https://github.com/orrzohar/PROB
6. https://github.com/DIG-Beihang/ALLOW
7. https://openaccess.thecvf.com/content/CVPR2023/papers/Ma_CAT_LoCalization_and_IdentificAtion_Cascade_Detection_Transformer_for_Open-World_Object_CVPR_2023_paper.pdf - https://github.com/Jq-F/CAT
8. https://github.com/Went-Liang/UnSniffer
9. https://github.com/boschresearch/Hyp-OW
10. https://proceedings.neurips.cc/paper_files/paper/2024/file/8766fbc68e1ed1cdef712ce273e0a363-Paper-Conference.pdf - https://github.com/xxyzll/UMB
11. https://neurips.cc/virtual/2023/poster/72249
12. https://arxiv.org/pdf/1909.13032
13. https://arxiv.org/pdf/2003.06957
14. https://cvpr.thecvf.com/virtual/2025/poster/32637
15. https://cvpr.thecvf.com/virtual/2025/poster/32726
16. https://arxiv.org/abs/2510.09173
17. https://www.cs.toronto.edu/~zemel/documents/prototypical_networks_nips_2017.pdf
