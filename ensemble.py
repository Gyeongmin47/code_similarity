import pandas as pd
import numpy as np

submission = pd.read_csv('./data/sample_submission.csv')

submission_1 = pd.read_csv('./data/submission_graphcodebert_BM25L_0610.csv')
submission_2 = pd.read_csv('./data/submission_CodeBERTaPy_0610.csv')
submission_3 = pd.read_csv('./data/submission_codebert_mlm_BM25L_0610.csv')

sub_1 = submission_1['similar']
sub_2 = submission_2['similar']
sub_3 = submission_3['similar']

# ensemble_preds = (sub_1 + sub_3) / 2    # 1, 0.5, 0.5, 0
ensemble_preds = (sub_1 + sub_2 + sub_3) / 3

# preds = np.where(ensemble_preds >= 0.5, 1, 0)
preds = np.where(ensemble_preds > 0.5, 1, 0)    # 두 모델 이상에서 맞다고 할 경우

submission['similar'] = preds

# submission.to_csv('./data/submission_ensemble_0610.csv', index=False)
submission.to_csv('./data/submission_ensemble_0610_v2.csv', index=False)