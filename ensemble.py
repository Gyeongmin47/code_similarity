import pandas as pd
import numpy as np

submission = pd.read_csv('./data/sample_submission.csv')

submission_1 = pd.read_csv('./data/submission.csv')
# submission_2 = pd.read_csv('./data/submission_CodeBERTaPy_2.csv')
submission_3 = pd.read_csv('./data/submission_codebert-mlm_1.csv')

sub_1 = submission_1['similar']
# sub_2 = submission_2['similar']
sub_3 = submission_3['similar']

# ensemble_preds = (sub_1 + sub_2 + sub_3) / 3
ensemble_preds = (sub_1 + sub_3) / 2

preds = np.where(ensemble_preds > 0.4, 1, 0)

submission['similar'] = preds

submission.to_csv('./data/submission_ensemble_v2.csv', index=False)