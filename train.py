import sklearn2pmml
import pandas as pd
import warnings
import numpy

from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

INPUT = 'train-data.csv'
LABEL_COL = 'deleted_at'
FEATURE_COLS = ['body']
# prediction output column
TARGET_COL = 'score'

warnings.filterwarnings('ignore')

# Load and split data
data = pd.read_csv(
    INPUT,
    usecols = [LABEL_COL] + FEATURE_COLS)
for column in FEATURE_COLS:
    data[column].fillna(value = '', inplace = True)
labels = (~data[LABEL_COL].isnull()).astype('int')
train_data, test_data, train_labels, test_labels = train_test_split(data['body'], labels, test_size = 0.1)

# Define the pipeline

pipeline = PMMLPipeline([
    ('tfidf', TfidfVectorizer(
        analyzer = 'word',
        stop_words = 'english',
        use_idf = True,
        sublinear_tf = True,
        max_df = 0.1,
        min_df = 0.0001,
        norm = None,
        tokenizer = Splitter()
    )),
    ('linear', SGDClassifier(
        # params
    )),


])
# Train the model
pipeline.fit(train_data, train_labels)

# Pack verification data and verify
pipeline.verify(test_data)

# Save the pipeline + model
sklearn2pmml.sklearn2pmml(sklearn2pmml.make_pmml_pipeline(
    pipeline,
    active_fields=FEATURE_COLS,
    target_fields=TARGET_COL
), 'model.pmml', with_repr=True, debug=True)

# Measure the accuracy
descision = pipeline.decision_function(test_data.sample(1))


jpmml_input_data = pd.read_csv(
    "jpmml-test-input.csv",
    usecols = FEATURE_COLS)
input_pred = pipeline.predict(jpmml_input_data.squeeze())
with numpy.printoptions(threshold=numpy.inf):
    print(input_pred)

pred = pipeline.predict(test_data)
print('Accuracy = {:.3f}'.format(sum(l == p for l, p in zip(test_labels, pred)) / len(test_labels)))
print('Dominant label freq = {:.3f}'.format(1 - sum(test_labels) / len(test_labels)))
print('ROC AUC = {:.3f}'.format(roc_auc_score(test_labels, pred)))
print(len(train_labels), 'training data rows')
print(len(pipeline[1].coef_[0]), 'model parameters')
