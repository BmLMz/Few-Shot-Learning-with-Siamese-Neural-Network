# Few-Shot-Learning-with-Siamese-Neural-Network
**Few-Shot classification on CIFAR100 classes with Siamese Neural Network in Pytorch**

This code is plug and play. For my Siamese Neural Network I used first a homemade simple Conv Neural NetWork and then a pretrained (on ImageNet) Resnet18. This small project is to explore classification in Siamese Neural Network. Of course Siamese Neural Networks are not suited for classification purposes, in our case we can see Siamese Neural Network as a clustering task mapping embeddings in a vector space considering instances with the same label closer and inscrease distance/similarity for instances with different label. The entire few-shot principle is relying on the fact that our Network descriptors are enough discriminant to automatically map far away emebeddings that do not belong to the same class even if classes have not been seen in the training phase. Siamese NN are well suited for similarity computations with an unlimited amount of classes, here we explore classification. I made a quick overview where I wanted to observe the behaviourof such networks on classification and Few-Shot.

## Usage
- Optimizer is Adam with learining rate 5e-4
- Loss could be chosen between constrastive loss or triplet loss
- Similarity function can be chosen between Euclidean distance and  Cosine similarity.
- Data as been extracted from CIFAR100 with cifar2png.

In this repository I used 6 classes: Apple, Chair, Wardrobe, Leopard, Clock and Tractor. This last class is the one used for Few-Shot. In order to re-use the code you need to format your data as following:
- data
  - glob
    - Subfolders with the name of your classes containing 1 instance of the class (the one you will compare to for classfication as Siamese NN will only compute similarity between 2 instances).
  - train
    - Subfolders with the name of you classes containing training instances
  - test
    - Subfolders with the name of you classes containing training instances
 
The subfolder glob contains one image for each class, that is for computing accuracy wich as we will assign a label according to the maximum similarity between the considered instance and the representatives of each class.
    
## Results 
The following results have been obtained ovr 10 iterations (reproducing the experience 10 times with random picks for few-shots instaces and for each few-shot number of shot), so that we can average performances. I used the 6 classes described above, with Resnet18 (pretrained), contrastive loss and euclidean distance:

| N-Shots | 0 | 1 | 3 | 5 |
|---|---|---|---|---|
| Global accuracy on test (%) | 88 | 88 | 89 | 89 |
| Few-Shot class accuracy (%) | 52 | 58 | 60 | 65 |

Results are not that bad, especially for very low-shot learning (0 and 1, by 0 I understand that I did not use instances of the few-class in the training phase).  things might be denoted here:
- Results are not that bad but I chose on purpose classes that are very dissimilar, so it obviously helps our algorithm
- Results have an high variance and convergence is really unstable. 

## References
- Harveyslash/Facial-Smilarity-with-Siamase-Networks  on <a href = 'https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch'> GitHub </a>
