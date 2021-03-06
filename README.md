# Anti-spoofing-model-based-on-AlexNet-and-Lcnn
The goal of this paper is to detect the replay speech and realize anti-spoofing modeling with two convolutional neural networks, namely, AlexNet(based on transfer learning) and LCNN model.

In this paper, the pre-trained AlexNet is trained twice, and the full connection layer is modified as a binary classification model. In order to compare the classification effect of AlexNet and SVM on the speech data set, a comparative experiment was conducted on AlexNet and AlexNet+SVM. The single AlexNet model processes the feature extraction and classification of speech signals together. In AlexNet+SVM, AlexNet is used as the front-end to extract features, and SVM classifies features. The experimental results show that AlexNet performs well in the detection of replay speech, and the classification effect is better than that of SVM. On the basis of these two experiments, the best experimental results were obtained by standardizing the input data of SVM, and the EER (equal-error rate) on the test set was 13.94%.

LCNN has five convolutional layers, four NIN layers and nine MFM layers. Flexible application of MFM, NIN and batch normalization layer is the main feature of LCNN. The combination of MFM and NIN realizes the data drop and channel convergence of the convolutional neural network, and the batch normalization makes the network converge faster. LCNN is a relatively lightweight convolutional neural network model compared with AlexNet. The total model parameter of LCNN is 41.9k, which is 0.31% of the latter. Under the condition of the same frequency resolution, the EER of LCNN on the test set is 14.54%, and that of AlexNet on the test set is 14.03%.

In this paper, the log energy spectrogram of STFT is processed in three ways: global normalization, local normalization, and non-normalization. All these were tested on LCNN. In general, data normalization can improve the effect of model training. In the comparison between global normalization and local normalization, the experimental results in this paper show that the latter is more suitable for STFT log energy spectrogram processing than the former. Therefore, the log energy spectra of STFT are processed in this way.

The main work of this paper is: 
1) The introduction the basic technology of extracting phonetic features.
2) AlexNet and LCNN models are used to detect the replay voice on ASVspoof 2017 data set.
3) The log energy spectra of STFT were processed in different normalized ways and used as the input data of LCNN for experiments.
4) The analyzation the training effect of the convolutional neural network model.

![image](https://github.com/RabbitSea/Anti-spoofing-model-based-on-AlexNet-and-Lcnn/blob/master/AlexNet.PNG)

![image](https://github.com/RabbitSea/Anti-spoofing-model-based-on-AlexNet-and-Lcnn/blob/master/Lcnn.PNG)
