# Anomaly-Detection-on-Dynamic-Graph-to-identify-disinformation

## Initial setting

### Downloads

Download the 'code' file of the present repository. Codes from TGN (https://github.com/twitter-research/tgn) and TADDY (https://github.com/yixinliu233/TADDY_pytorch) repositories are reused here for the sake of compatibility.
Compare to the original repositories, the following files were replaced or added:
 - For TGN:
   - 'train_self_supervised.py';
   - 'utils/preprocess_data.py';
   - 'utils/data_processing.py';
   - 'model/tgn.py';
   - 'evaluation/evaluation.py';
   - 'modules/embedding_module.py';
   - 'TGN_TADDY.ipynb'.
 - For TADDY:
   - '0_prepare_data.py';
   - '1_train.py';
   - 'codes/AnomalyGeneration.py';
   - 'data/raw/email-dnc.csv';
   - 'data/raw/AST'.

## Running the codes

Open the code/TGN/Main_notebook.ipynb notebook. Run all to test it on the btc_alpha benchmark dataset with 5% of anomalies. Results are available in Part. C.3) and Part. D.2) of the notebook.

If you want to try another benchmark dataset, three cells need to be updated :
 - In Part. A.1)a-: choice of the dataset, you need to indicate two times the name of the chosen dataset (uci, digg, email, btc_alpha, btc_otc or AST);
 - In Part B.1): for benchmark datasets, you need to indicate the name of the chosen dataset and the anomaly proportion with a specific format, e.g. for btc_alpha and 5% of anomalies: btc_alpha_TADDY_005_nop, you need also to precise the last instant of training obtained in at the end of Part. A.1) (otherwise, some of these values are given at the end of this readme);
 - In Part C.1): choice of the dataset, you need to specify the chosen dataset with the aforementionned format.

If you want to test the method with the Synthetic dataset, you need to decomment the cells allowing its generation in Part. A.2), the training of TGN on it in Part. B.2) and to use 'Synthetic' as dataset name everywhere.

## Last training instant for benchmark datasets

 - uci: 2946672.0
 - email: 82975888.0
 - btc_alpha: 57711600.0
 - btc_otc: 70837287.36343002
