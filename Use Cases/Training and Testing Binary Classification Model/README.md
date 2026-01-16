It is originally from the UGR'16 network dataset.
We did some modification to select some datset samples to demo the use case of augmenting the training data on downstream binary classification.
* We have source IP address and destination IP address split into four 8-bit integer columns for each.
* Select `label` column with two classes: background -- benign network flow, blacklist --unbenign network flow

The training dataset is imbalanced: 20,000 backgrounds and 4000 blacklists, sampling from April dataset (80%)
The testing dataset is balanced: 3000 backgrounds and 3000 blacklists, sampling from May dataset (20%)