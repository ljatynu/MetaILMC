# MetaILMC
# MetaPKI: Meta-learning based Inductive Logistic Matrix Completion for Prediction of Kinase Inhibitors

This code is an implementation of our paper
" MetaPKI: Meta-learning based Inductive Logistic Matrix Completion for Prediction of Kinase Inhibitors" in PyTorch. In this repository, we provide two Kinase datasets: KinaseDB and LTKinaseDB.

In our problem setting of MetaPKI prediction, the input of our model is the pair of a SMILES format of compound and an
amino acid sequence of protein and the outputs is a binary prediction result interactionprediction result. 

The pseudocode of our MetaPKI Model is as follows:

![image-20221130200803853](https://dum1615.oss-cn-chengdu.aliyuncs.com/image-20221130200803853.png)

The details of the MetaPKI Model are described in our paper.

## Characteristics of the repository

* We provide the **several demo-scripts** of the experiments in the paper, so that you can quickly understand the prediction process of the MetaPKI model.
* This code is **easy to use**. It means that you can customize your own dataset and train your own MetaPKI prediction
  model, and apply it to the new "drug discovery" scenario.

## Requirements

- Pytorch 1.6.0
- Numpy 1.19.5
- Scikit-learn 0.23.2
- RDKit 2022.3.1b1
- Pandas 1.1.5

## (a) Run ILMC Model

We provide two ways to run the ILMC model

1. Verify the global predictive performance with 10-Fold-Cross Validation

   Run the script "ILMC/main.py" with KinaseDB dataset to verify the global predictive performance with ILMC model(For the detailed description of this experiment, please refer to **4.3 Predictive Performance of ILMC**  in the paper). Experiments result as following:

   ```
      | FOLD | AUC     | BA     | AUPR     | F1     | RECALL     | PRECISION     |
      | ---- | ------- | ------ | -------- | ------ | ---------- | ------------- |
      | 0    | 0.9173  | 0.8262 | 0.8479   | 0.7642 | 0.7821     | 0.8047        |
      | 1    | 0.9211  | 0.8349 | 0.8491   | 0.7724 | 0.7913     | 0.8076        |
      | 2    | 0.9186  | 0.8325 | 0.8482   | 0.774  | 0.7877     | 0.808         |
      | 3    | 0.9196  | 0.8336 | 0.8502   | 0.7727 | 0.7835     | 0.8033        |
      | 4    | 0.9181  | 0.8306 | 0.851    | 0.7701 | 0.7735     | 0.8178        |
      | 5    | 0.9184  | 0.8312 | 0.8425   | 0.7653 | 0.7794     | 0.8017        |
      | 6    | 0.9192  | 0.8292 | 0.854    | 0.7696 | 0.7741     | 0.8102        |
      | 7    | 0.9203  | 0.8292 | 0.8527   | 0.7692 | 0.7757     | 0.8091        |
      | 8    | 0.9198  | 0.8284 | 0.8509   | 0.7693 | 0.7806     | 0.8178        |
      | 9    | 0.9183  | 0.8285 | 0.8498   | 0.7686 | 0.7729     | 0.8099        |
   ```

2. Verify the predictive performance with few-shot on LTKinaseDB

   Run the script "metaILMC/main.py" with LTKinaseDB dataset to verify the predictive performance on tail tasks, this script will calculate AUC, AUPR, F1-score, PRECISION, BA(balance accuracy) and RECALL in each tail tasks. Experiments result as following:

   ```
      | UniportID   | AUC    | AUPR   | F1     | PRECISION | BA     | RECALL |
      | ----------- | ------ | ------ | ------ | --------- | ------ | ------ |
      | Q8IVH8      | 0.8857 | 0.808  | 0.7059 | 0.6       | 0.8486 | 0.8571 |
      | O15146      | 0.875  | 0.44   | 0.6667 | 0.5       | 0.8929 | 1      |
      | Q16816      | 0.9209 | 0.8202 | 0.8764 | 0.78      | 0.8942 | 1      |
      | Q15375      | 0.7344 | 0.5402 | 0.2857 | 0.2       | 0.7344 | 1      |
      | P21803      | 0.8846 | 0.7643 | 0.8    | 0.75      | 0.8846 | 1      |
      | P29317      | 0.8497 | 0.5119 | 0.5097 | 0.3977    | 0.7736 | 0.8716 |
      | P48729      | 0.686  | 0.2501 | 0.3388 | 0.2464    | 0.6464 | 0.6794 |
      | P41240      | 0.8487 | 0.5415 | 0.4314 | 0.2821    | 0.7525 | 0.9167 |
      | P90584      | 0.7444 | 0.1553 | 0.2174 | 0.1333    | 0.7012 | 0.7143 |
      | P35916      | 0.859  | 0.7912 | 0.6685 | 0.6872    | 0.7619 | 0.8495 |
      | Q64725      | 0.7403 | 0.4189 | 0.4569 | 0.3966    | 0.6911 | 0.7708 |
      | Q9NR20      | 0.7779 | 0.171  | 0.2436 | 0.1515    | 0.7483 | 0.7778 |
      | Q9Y463      | 0.7361 | 0.5564 | 0.5234 | 0.5556    | 0.6638 | 0.5528 |
      | Q8IU85      | 0.6986 | 0.352  | 0.3721 | 0.3478    | 0.6632 | 0.625  |
      | P05131      | 0.8256 | 0.5892 | 0.4333 | 0.3       | 0.7832 | 0.9375 |
      | Q9WUD9      | 0.95   | 0.9727 | 0.9583 | 0.9583    | 0.9292 | 0.9583 |
      | P17948      | 0.6484 | 0.6284 | 0.6262 | 0.6711    | 0.6203 | 0.7543 |
   	………………
      | Q02779      | 0.8427 | 0.689  | 0.6341 | 0.5652    | 0.7842 | 0.9444 |
      | Q13555      | 0.7506 | 0.3188 | 0.3585 | 0.2565    | 0.6931 | 0.8791 |
      | Q9NSY1      | 0.9556 | 0.8211 | 0.6667 | 0.5714    | 0.9074 | 1      |
      | Q15303      | 0.8449 | 0.4981 | 0.46   | 0.386     | 0.7546 | 0.8481 |
      | P49137      | 0.6461 | 0.4558 | 0.527  | 0.4643    | 0.6216 | 0.8218 |
      | P07949      | 0.756  | 0.6198 | 0.623  | 0.6126    | 0.6977 | 0.7447 |
      | O94804      | 0.7786 | 0.6997 | 0.7333 | 0.6875    | 0.7679 | 0.9286 |
      | Q16620      | 0.8685 | 0.5843 | 0.5902 | 0.5172    | 0.7799 | 0.8588 |
      | Q7KZI7      | 0.8212 | 0.4535 | 0.4639 | 0.3431    | 0.7514 | 0.8667 |
      | P00517      | 0.7363 | 0.3649 | 0.4719 | 0.4118    | 0.7037 | 0.8387 |
      | Q9HCP0      | 0.6884 | 0.2775 | 0.3192 | 0.2237    | 0.6333 | 0.8068 |
      | Q02750      | 0.5941 | 0.7192 | 0.6127 | 0.702     | 0.5851 | 0.5489 |
      | P53667      | 0.6773 | 0.2801 | 0.3764 | 0.3258    | 0.6467 | 0.6535 |
      | Q9UEE5      | 0.7393 | 0.5948 | 0.5688 | 0.6115    | 0.6791 | 0.8333 |
      | P49674      | 0.5217 | 0.3259 | 0.35   | 0.2609    | 0.5652 | 0.75   |
      | Q06418      | 0.7697 | 0.3913 | 0.3791 | 0.3019    | 0.6867 | 0.7549 |
      | P78368      | 0.6958 | 0.2966 | 0.34   | 0.2431    | 0.6468 | 0.8409 |
      | P54764      | 1      | 1      | 0.2    | 0.1111    | 0.8889 | 1      |
      | P05622      | 0.6593 | 0.483  | 0.482  | 0.4269    | 0.6218 | 0.6835 |
      | P43405      | 0.6122 | 0.6535 | 0.5976 | 0.586     | 0.5748 | 0.7294 |
      | P30291      | 0.6931 | 0.7834 | 0.8296 | 0.824     | 0.7237 | 0.8602 |
      | Q15831      | 0.75   | 0.0556 | 0.1818 | 0.1       | 0.8594 | 1      |
      | P09215      | 0.801  | 0.4667 | 0.5228 | 0.3815    | 0.7533 | 0.8919 |
      | O14578      | 0.5385 | 0.323  | 0.375  | 0.3333    | 0.6154 | 0.8333 |
      | P49759      | 0.7368 | 0.6576 | 0.7273 | 0.6765    | 0.7118 | 0.8421 |
      | Q9NYY3      | 0.7719 | 0.6735 | 0.5    | 1         | 0.6586 | 0.8947 |
      | O15264      | 0.6797 | 0.6371 | 0.7432 | 0.6751    | 0.6676 | 0.872  |
      | Q13043      | 0.9009 | 0.848  | 0.7742 | 0.8333    | 0.8556 | 0.9333 |
   ```

   

## (b) Run MetaILMC Model

Run the script "metaILMC/main.py" with LTKinaseDB dataset to verify MetaILMC predictive performance on tail tasks, meta-training phase use head tasks and meta-testing phase use tail tasks, this script will calculate AUC, AUPR, F1-score, PRECISION, BA(balance accuracy) and RECALL in each tail tasks. Experiments result as following:

```
| Uniport ID  | AUC    | AUPR   | F1     | PRECISION | BA     | RECALL |
| ----------- | ------ | ------ | ------ | --------- | ------ | ------ |
| Q8IVH8      | 1      | 1      | 0.8    | 0.6667    | 0.9464 | 1      |
| O15146      | 0.9702 | 0.9658 | 0.9024 | 0.9062    | 0.9167 | 1      |
| Q16816      | 0.8906 | 0.5705 | 0.3077 | 0.2       | 0.8594 | 1      |
| Q15375      | 1      | 1      | 1      | 1         | 1      | 1      |
| P21803      | 0.8818 | 0.6126 | 0.5952 | 0.5245    | 0.8249 | 0.9266 |
| P29317      | 0.7772 | 0.434  | 0.4421 | 0.3663    | 0.7245 | 0.8931 |
| P48729      | 0.8388 | 0.6256 | 0.5263 | 0.3913    | 0.8088 | 1      |
| P41240      | 0.7038 | 0.2593 | 0.2    | 0.1163    | 0.7038 | 0.8571 |
| P90584      | 0.9013 | 0.8659 | 0.7856 | 0.8551    | 0.8454 | 0.913  |
| P35916      | 0.753  | 0.4697 | 0.5    | 0.4694    | 0.7204 | 0.7708 |
| Q64725      | 0.8284 | 0.2774 | 0.2635 | 0.1571    | 0.7906 | 1      |
| Q9NR20      | 0.8054 | 0.607  | 0.6255 | 0.5882    | 0.741  | 0.8049 |
| Q9Y463      | 0.8163 | 0.405  | 0.4252 | 0.35      | 0.7735 | 0.85   |
| Q8IU85      | 0.8261 | 0.3538 | 0.4074 | 0.3333    | 0.7634 | 0.9375 |
| P05131      | 0.975  | 0.9886 | 0.9583 | 1         | 0.9292 | 1      |
| Q9WUD9      | 0.7339 | 0.7444 | 0.7293 | 0.7228    | 0.6734 | 0.9429 |
| P17948      | 0.8043 | 0.5583 | 0.5882 | 1         | 0.7255 | 0.875  |
…………
| P11440      | 0.838  | 0.6387 | 0.6264 | 0.76      | 0.7842 | 1      |
| Q16512      | 1      | 1      | 0.8571 | 1         | 0.9839 | 1      |
| Q9Y243      | 0.9224 | 0.6946 | 0.6326 | 0.6196    | 0.8759 | 1      |
| Q13188      | 0.8244 | 0.6328 | 0.6101 | 0.6536    | 0.76   | 0.9279 |
| P68400      | 0.7226 | 0.4624 | 0.4282 | 0.4103    | 0.6708 | 0.8187 |
| Q07912      | 0.8609 | 0.7638 | 0.692  | 0.7143    | 0.7929 | 0.9333 |
| P49187      | 0.7652 | 0.805  | 0.7536 | 1         | 0.7429 | 1      |
| Q16513      | 0.8336 | 0.6593 | 0.6045 | 0.6185    | 0.7578 | 0.8785 |
| P53778      | 0.7893 | 0.7988 | 0.7675 | 0.9024    | 0.7244 | 0.9573 |
| P54753      | 0.8644 | 0.6446 | 0.6429 | 0.5625    | 0.8559 | 1      |
| Q13164      | 0.9579 | 0.8786 | 0.8    | 0.7059    | 0.9103 | 1      |
| Q02779      | 0.9786 | 0.9573 | 0.7727 | 0.6538    | 0.9077 | 1      |
| Q13555      | 0.8573 | 0.5836 | 0.4667 | 0.4068    | 0.7771 | 0.978  |
| Q9NSY1      | 1      | 1      | 1      | 1         | 1      | 1      |
| Q15303      | 0.8443 | 0.5866 | 0.5542 | 0.55      | 0.7873 | 0.962  |
| P49137      | 0.7483 | 0.5931 | 0.5784 | 0.4865    | 0.6848 | 0.9604 |
| P07949      | 0.782  | 0.6815 | 0.6469 | 0.6462    | 0.7115 | 0.8829 |
| O94804      | 0.8786 | 0.855  | 0.8148 | 1         | 0.8429 | 1      |
| Q16620      | 0.844  | 0.5876 | 0.5486 | 0.4916    | 0.7773 | 0.9435 |
| Q7KZI7      | 0.8583 | 0.592  | 0.5015 | 0.411     | 0.783  | 0.9905 |
| P00517      | 0.8448 | 0.5964 | 0.5672 | 0.5278    | 0.7728 | 1      |
| Q9HCP0      | 0.7134 | 0.3266 | 0.3719 | 0.269     | 0.6786 | 0.8864 |
| Q02750      | 0.7322 | 0.8004 | 0.7566 | 0.8333    | 0.6829 | 0.836  |
| P53667      | 0.7549 | 0.423  | 0.4143 | 0.3243    | 0.7009 | 0.937  |
| Q9UEE5      | 0.7916 | 0.6696 | 0.6105 | 0.5679    | 0.7163 | 0.9635 |
| P49674      | 0.8134 | 0.6788 | 0.6667 | 0.5833    | 0.8297 | 1      |
| Q06418      | 0.8255 | 0.5314 | 0.4667 | 0.3984    | 0.7666 | 0.9412 |
| P78368      | 0.7675 | 0.4012 | 0.43   | 0.3014    | 0.7391 | 0.9545 |
| P54764      | 1      | 1      | 0.4    | 0.25      | 0.9583 | 1      |
| P05622      | 0.7468 | 0.5193 | 0.5405 | 0.4863    | 0.6749 | 1      |
| P43405      | 0.7458 | 0.7628 | 0.7189 | 0.708     | 0.6894 | 0.8824 |
| P30291      | 0.86   | 0.9281 | 0.881  | 0.875     | 0.7674 | 1      |
| Q15831      | 1      | 1      | 1      | 1         | 1      | 1      |
| P09215      | 0.8682 | 0.6389 | 0.6154 | 0.5583    | 0.7977 | 0.9892 |
| O14578      | 0.9679 | 0.9205 | 0.8333 | 1         | 0.9423 | 1      |
| P49759      | 0.9486 | 0.9459 | 0.8916 | 1         | 0.8938 | 1      |
| Q9NYY3      | 0.915  | 0.8513 | 0.7917 | 1         | 0.8718 | 1      |
| O15264      | 0.8136 | 0.8411 | 0.7911 | 0.9565    | 0.7456 | 0.9858 |
| Q13043      | 0.9787 | 0.9549 | 0.9333 | 1         | 0.9583 | 1      |
```



## (c) Case Study

when you run "metaILMC/main.py" script, it will save best model parameters for each tail tasks to "metaILMC/model", thus you can use these model parameters to verify new drugs predictive performance on tail tasks, we select two drugs which not in our datasets to predictive performance. Experiment result as following:

* Dasatinib

  ```
    | UniPort ID  | Name      | Active points | Inactive Points | Probability |
    | ----------- | --------- | ------------- | --------------- | ----------- |
    | P06494      | rERBB2    | 64            | 525             | 0.9983      |
    | Q9Y4K4      | hMAP4K5   | 257           | 457             | 0.998       |
    | Q8N4C8      | hMINK1    | 170           | 493             | 0.9965      |
    | P07948      | hLYN      | 364           | 583             | 0.996       |
    | P05480      | mSRC      | 146           | 554             | 0.995       |
    | P05696      | rPKCa     | 116           | 135             | 0.9941      |
    | Q92918      | hMAP4K1   | 12            | 29              | 0.9928      |
    | P54753      | hEPHB3    | 17            | 64              | 0.9919      |
    | P36896      | hALK4     | 74            | 316             | 0.9917      |
    | P48730      | hCSNK1D   | 179           | 673             | 0.991       |
    | P54760      | hEPHB4    | 192           | 277             | 0.9902      |
    | P51451      | hBLK      | 244           | 463             | 0.9871      |
    | P43403      | hZAP70    | 85            | 149             | 0.9865      |
    | P22455      | hFGFR4    | 30            | 111             | 0.9853      |
    | Q13882      | hPTK6     | 44            | 408             | 0.9832      |
    | P54756      | hEPHA5    | 8             | 36              | 0.9789      |
    | P14616      | hINSRR    | 84            | 688             | 0.9752      |
    …………
    | P17948      | hVEGFR1   | 408           | 367             | 0.0163      |
    | Q05397      | hFAK1     | 204           | 806             | 0.0147      |
    | Q14164      | hIKBKE    | 54            | 323             | 0.0139      |
    | Q16816      | hPHKG1    | 44            | 57              | 0.0138      |
    | Q5VT25      | hCDC42BPA | 86            | 722             | 0.0129      |
    | O43781      | hDYRK3    | 110           | 418             | 0.012       |
    | P51957      | hNEK4     | 68            | 530             | 0.0116      |
    | P29376      | hLTK      | 116           | 271             | 0.0106      |
    | Q96SB4      | hSRPK1    | 49            | 608             | 0.009       |
    | P36897      | hTGFbR1   | 319           | 168             | 0.0081      |
    | P78368      | hCSNK1G2  | 93            | 568             | 0.0079      |
    | P35969      | mVGFR1    | 545           | 320             | 0.0073      |
    | P51817      | hPRKX     | 205           | 611             | 0.0069      |
    | Q00534      | hCDK6     | 12            | 56              | 0.0067      |
    | Q9Y616      | hIRAK3    | 6             | 35              | 0.0064      |
    | Q9UIK4      | hDAPK2    | 6             | 40              | 0.0062      |
    | P49761      | hCLK3     | 7             | 70              | 0.0062      |
    | Q9Y463      | hDYRK1B   | 128           | 309             | 0.0042      |
    | P48729      | hCSNK1A1  | 136           | 863             | 0.004       |
    | P34947      | hGRK5     | 40            | 310             | 0.0038      |
    | P11440      | mCDC2     | 176           | 650             | 0.0035      |
    | P50613      | hCDK7     | 158           | 397             | 0.0023      |
    | Q02763      | hTIE2     | 417           | 319             | 0.0022      |
    | Q00537      | hPCTK2    | 7             | 35              | 0.002       |
    | XP_341554.1 | rPKCt     | 157           | 382             | 0.0019      |
    | P80192      | hMAP3K9   | 44            | 47              | 0.0015      |
    | P22607      | hFGFR3    | 42            | 120             | 0.0014      |
    | Q13237      | hPRKG2    | 81            | 494             | 0.0012      |
    | Q13131      | hPRKAA1   | 166           | 754             | 0.0003      |
    | P90584      | plPfmrk   | 12            | 114             | 0.0001      |
  ```

* Sunitinib

  ```
  | UniPort ID  | Name      | Active points | Inactive Points | Probability |
    | ----------- | --------- | ------------- | --------------- | ----------- |
    | P35969      | mVGFR1    | 545           | 320             | 1           |
    | O95819      | hMAP4K4   | 366           | 672             | 0.9997      |
    | Q96SB4      | hSRPK1    | 49            | 608             | 0.9997      |
    | P23443      | hRPS6KB1  | 291           | 568             | 0.9996      |
    | O15530      | hPDPK1    | 226           | 782             | 0.9996      |
    | O00444      | hPLK4     | 215           | 557             | 0.9996      |
    | Q9H2X6      | hHIPK2    | 208           | 486             | 0.9994      |
    | O94806      | hPKD3     | 294           | 1008            | 0.9992      |
    | P52333      | hJAK3     | 461           | 948             | 0.999       |
    | Q03142      | mFGFR4    | 11            | 18              | 0.9987      |
    | P22607      | hFGFR3    | 42            | 120             | 0.9987      |
    | O60285      | hNUAK1    | 10            | 55              | 0.9986      |
    | Q9BZL6      | hPKD2     | 136           | 663             | 0.9983      |
    | Q08881      | hITK      | 455           | 704             | 0.9983      |
    | Q96RG2      | hPASK     | 16            | 64              | 0.9983      |
    | Q92918      | hMAP4K1   | 12            | 29              | 0.9977      |
    | Q13554      | hCAMK2B   | 61            | 702             | 0.9974      |
    …………
    | P35590      | hTIE1     | 13            | 28              | 0.1051      |
    | Q00535      | hCDK5     | 610           | 1353            | 0.0978      |
    | Q86UE8      | hTLK2     | 5             | 37              | 0.0907      |
    | Q15208      | hSTK38    | 2             | 5               | 0.0726      |
    | P70618      | rp38a     | 45            | 8               | 0.0663      |
    | Q8WTQ7      | hGRK7     | 5             | 3               | 0.0643      |
    | Q15835      | hGRK1     | 5             | 4               | 0.0641      |
    | Q15303      | hHER4     | 84            | 653             | 0.0624      |
    | P27361      | hERK1     | 84            | 315             | 0.0591      |
    | Q86UX6      | hSTK32C   | 1             | 40              | 0.0502      |
    | P49841      | hGSK3b    | 1351          | 1550            | 0.04        |
    | Q00537      | hPCTK2    | 7             | 35              | 0.0347      |
    | P42681      | hTXK      | 15            | 39              | 0.0329      |
    | Q00534      | hCDK6     | 12            | 56              | 0.0321      |
    | P18266      | rGSK3b    | 5             | 1               | 0.0309      |
    | P42679      | hMATK     | 31            | 434             | 0.0175      |
    | P11440      | mCDC2     | 176           | 650             | 0.0151      |
    | P53779      | hJNK3     | 345           | 735             | 0.01        |
    | P41241      | mCSK      | 57            | 28              | 0.0092      |
    | P90584      | plPfmrk   | 12            | 114             | 0.0046      |
  ```

  

 ## (d) Parameters Selection

In the ablation experiment in our paper(please refer to **4.5 Effect of Parameter Setting on MetaILMC Prediction Performance**  in the paper for the details). We found that when number of meta-training tasks increase, the predictive performance on tail tasks will show better. Beyond that, the inner-loop learning rate 𝛼, the gradient descent steps of inner-loops, and the outer-loop learning rate 𝛽 all affect both generalization and convergence speed of MetaILMC, we select the hyperparameters through experiments.  You can also try to adjust these hyperparameters based on your own dataset to achieve better performance. 



## (e) Training of MetaILMC Model with your dataset

**We recommend you to run "/metaILMC/main.py" script to reproduce our experiment before attempting to train the MetaILMC model
using your own dataset to familiarize yourself with the training and prediction processes of the MetaILMC.**

**- Step-1: Raw data format**

The dataset should contain kinase sequences, compound smiles sequences and label between kinases and compounds, examples of compound SMILES sequences and kinase sequences are as follows:

```
compound smiles:
c1ccc(c(c1)c2cccc3c2ccc(=O)n3c4c(cccc4Cl)Cl)Cl
c1cc(cc(c1)F)COc2ccc(cc2Cl)Nc3c4c(cc(s4)c5ccc[nH]5)ncn3
CC(=O)N[C@H]1C[C@H](C1)n2cc(nc2)NC(=O)Cc3cccc4c3cccc4
c1ccc(cc1)c2ccc3c(c2)c(n[nH]3)N
CS(=O)(=O)Nc1ccc(cc1)Nc2cc([nH]n2)c3cccc(c3)Br
CSC1=Nc2ccccc2-c3c(c4cc(ccc4[nH]3)Br)C1
Cn1cc-2c(n1)CCc3c2c4c(c5c3[nH]c6c5cccc6)CNC4=O
CCn1c2cc(c(cc2c(=O)c3c1n(oc3=O)Cc4ccccc4)F)Cl
Cc1ccc(cc1)C(=O)/C=C/2\c3cc(ccc3NC2=O)Br
c1ccc(cc1)c2cc3ncc4c(n3n2)-c5c(cccn5)NC(=O)C4
c1cc(cc(c1)C#N)Cn2cnc(c2c3ccc(cc3)F)c4ccnc(n4)N

kinase sequences:
MRHSKRTHCPDWDSRESWGHESYRGSHKRKRRSHSSTQENRHCKPHHQFKESDCHYLEARSLNERDYRDRRYVDEYRNDYCEGYVPRHYHRDIESGYRIHCSKSSVRSRRSSPKRKRNRHCSSHQSRSKSHRRKRSRSIEDDEEGHLICQSGDVLRARYEIVDTLGEGAFGKVVECIDHGMDGMHVAVKIVKNVGRYREAARSEIQVLEHLNSTDPNSVFRCVQMLEWFDHHGHVCIVFELLGLSTYDFIKENSFLPFQIDHIRQMAYQICQSINFLHHNKLTHTDLKPENILFVKSDYVVKYNSKMKRDERTLKNTDIKVVDFGSATYDDEHHSTLVSTRHYRAPEVILALGWSQPCDVWSIGCILIEYYLGFTVFQTHDSKEHLAMMERILGPIPQHMIQKTRKRKYFHHNQLDWDEHSSAGRYVRRRCKPLKEFMLCHDEEHEKLFDLVRRMLEYDPTQRITLDEALQHPFFDLLKKK
MNPGFDLSRRNPQEDFELIQRIGSGTYGDVYKARNVNTGELAAIKVIKLEPGEDFAVVQQEIIMMKDCKHPNIVAYFGSYLRRDKLWICMEFCGGGSLQDIYHVTGPLSELQIAYVSRETLQGLYYLHSKGKMHRDIKGANILLTDNGHVKLADFGVSAQITATIAKRKSFIGTPYWMAPEVAAVERKGGYNQLCDLWAVGITAIELAELQPPMFDLHPMRALFLMTKSNFQPPKLKDKMKWSNSFHHFVKMALTKNPKKRPTAEKLLQHPFVTQHLTRSLAIELLDKVNNPDHSTYHDFDDDDPEPLVAVPHRIHSTSRNVREEKTRSEITFGQVKFDPPLRKETEPHHELPDSDGFLDSSEEIYYTARSNLDLQLEYGQGHQGGYFLGANKSLLKSVEEELHQRGHVAHLEDDEGDDDESKHSTLKAKIPPPLPPKPKSIFIPQEMHSTEDENQGTIKRCPMSGSPAKPSQVPPRPPPPRLPPHKPVALGNGMSSFQLNGERDGSLCQQQNEHRGTNLSRKEKKDVPKPISNGLPPTPKVHMGACFSKVFNGCPLKIHCASSWINPDTRDQYLIFGAEEGIYTLNLNELHETSMEQLFPRRCTWLYVMNNCLLSISGKASQLYSHNLPGLFDYARQMQKLPVAIPAHKLPDRILPRKFSVSAKIPETKWCQKCCVVRNPYTGHKYLCGALQTSIVLLEWVEPMQKFMLIKHIDFPIPCPLRMFEMLVVPEQEYPLVCVGVSRGRDFNQVVRFETVNPNSTSSWFTESDTPQTNVTHVTQLERDTILVCLDCCIKIVNLQGRLKSSRKLSSELTFDFQIESIVCLQDSVLAFWKHGMQGRSFRSNEVTQEISDSTRIFRLLGSDRVVVLESRPTDNPTANSNLYILAGHENSY
```



**- Step-2: Data preprocessing**

Then you need convert compound SMILE sequences and kinase sequences into feature embedding. We obtained the molecular fingerprint of the compounds using RDKit, and use the Conjoint Triad Descriptors method to get kinases feature embedding, see **4.1 Experimental Setup** in our paper for details.

```
compoung embedding:
[1	0	1	0	0	0	1	0	1	0	1	0……]
[0	0	0	1	0	0	0	0	1	0	1	0……]
[0	0	0	0	0	0	0	0	0	0	0	0……]
[0	1	0	0	0	0	0	1	0	0	1	0……]
[0	0	1	0	1	0	0	0	0	0	0	0……]
[0	0	0	0	0	1	0	0	1	0	0	0……]

kinase embedding:
[0.421	0.421	0.316	0.263	0.105	0.105……]
[0.667	0.5		0.333	0.056	0.333	0.222……]
[0.933	0.467	0.933	0.267	0.333	0.2……]
[0.571	0.143	0.286	0.143	0.286	0……]
[1		0.625	0.375	0.125	0.125	0.125……]
[0.944	0.611	0.278	0.333	0.278	0.056……]
```



**- Step-3: Encapsulate your own torch dataset**

You need to store the kinase and compound feature embedding into two csv format files, with the first column as its id,  and then use a csv file which format is  <compound id, kinase id, label> to store the data which will be trained. Run the script "metaILM/loadData.py" can load data to pytorch, actually, when you run "main.py" script, "loadData" will be called.

**- Step-4: Training**

Run "metaILMC/main.py" script, it willl load dataset and train MetaILMC model,  and save best model parameters for each tail tasks to "metaILMC/model"

**- Step-5: New Drug discovery**

After training, you can get best model parameters for each tail tasks, these parameters can performance good on tail tasks, and you can try to input new drug embedding into these models, to see if it has the potential to be a kinase inhibitor.

## Disclaimer

Please manually verify the reliability of the results by experts before conducting further drug experiments. Do not
directly use these drugs for disease treatment.

## Cite Us

If you found this work useful to you, please our paper:

```
@article{XXX,
  title={Meta-learning based Inductive Logistic Matrix Completion for Prediction of Kinase Inhibitors},
  author={XXX},
  journal={XXX},
  year={2022}
}
```



