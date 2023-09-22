import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow_text
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# The 16-language multilingual module is the default
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
use_multi_model = hub.load(module_url)
bert_multi_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


def process_tfidf_similarity(base_document, documents):
    vectorizer = TfidfVectorizer()
    # To make uniformed vectors, both documents need to be combined first.
    documents.insert(0, base_document)
    embeddings = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    return cosine_similarities[0]


def embed_text(input):
  return use_multi_model(input)

def process_use_similarity(base_document, ref_document):
	embeddings_1 = embed_text(base_document)
	embeddings_2 = embed_text(ref_document)
	sim = cosine_similarity(embeddings_1, embeddings_2)
	return sim[0][0]

def process_bert_similarity(base_document, documents):
	query_embedding = bert_multi_model.encode(text1)
	passage_embedding = bert_multi_model.encode(documents)
	sim = util.dot_score(query_embedding, passage_embedding).item()
	return sim



#####################
calculate_similarities = sys.argv[1]
processed_csv_path = "./similarity_with_translated.csv"
if calculate_similarities:
	translated_data_path = "./translated.csv"
	df = pd.read_csv(translated_data_path)

	cos_similarities = []
	use_similarities = []
	bert_similarities = []
	for text1, text2 in zip(df['text_1'], df['translated_text2']):
		cos_sim = process_tfidf_similarity(text1, [text2])
		use_sim = process_use_similarity(text1, text2)
		bert_sim = process_bert_similarity(text1, [text2])
		cos_similarities.append(cos_sim)
		use_similarities.append(use_sim)
		bert_similarities.append(bert_sim)

	# arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
	arccos_similarities = 1 - np.arccos(np.array(cos_similarities))/np.pi

	#####################
	df = df.assign(cos_similarity=cos_similarities)
	df = df.assign(arccos_similarity=arccos_similarities)
	df = df.assign(use_similarity=use_similarities)
	df = df.assign(bert_similarity=bert_similarities)
	df['arccos_similarity'] = df['arccos_similarity'].fillna(0)
	df.to_csv(processed_csv_path)
else:
	df = pd.read_csv(processed_csv_path)
# x = df[['Geography','Entities','Time','Narrative','Style','Tone', 'text_1', 'text_2', 'combined_text']]
x = df.drop(columns='Overall')
y = df['Overall']

# cos_pearsonr = stats.pearsonr(cos_similarities, y)
# arccos_pearsonr = stats.pearsonr(arccos_similarities, y)
# use_pearsonr = stats.pearsonr(use_similarities, y)
# bert_pearsonr = stats.pearsonr(bert_similarities, y)
print("Similarity Pearson correltion")
df[['cos_similarity', 'arccos_similarity', 'use_similarity', 'bert_similarity', 'Overall']].corr(method='pearson')
# print(cos_pearsonr, arccos_pearsonr, use_pearsonr, bert_pearsonr)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

####### Original Data
x_train_sims_only = x_train[['Geography', 'Entities', 'Time', 'Narrative', 'Style', 'Tone']]
x_test_sims_only = x_test[['Geography', 'Entities', 'Time', 'Narrative', 'Style', 'Tone']]


model_sims_only = LogisticRegression(max_iter=1000)
model_sims_only.fit(x_train_sims_only, y_train.round().astype('int'))

# predictions = model.predict(scaler.transform(x_test))
predictions = model_sims_only.predict(x_test_sims_only)
res_sims_only = classification_report(y_test.round().astype('int'), predictions)
print("results sims_only")
print(res_sims_only)
lr_sims_pearsonr = stats.pearsonr(y_test, predictions)
print("lr_sims_pearsonr: ", lr_sims_pearsonr)

x_train_text_only = x_train['combined_text']
x_test_text_only = x_test['combined_text']

vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word', ngram_range=(1,3),dtype=np.float32)
vect_char = TfidfVectorizer(max_features=40000, lowercase=True, analyzer='char', ngram_range=(3,6),dtype=np.float32)
tr_vect = vect_word.fit_transform(x_train_text_only)
ts_vect = vect_word.transform(x_test_text_only)

# Character n gram vector
tr_vect_char = vect_char.fit_transform(x_train_text_only)
ts_vect_char = vect_char.transform(x_test_text_only)

X_text = sparse.hstack([tr_vect, tr_vect_char])
X_test_text = sparse.hstack([ts_vect, ts_vect_char])

model_text_only = LogisticRegression(max_iter=1000)
model_text_only.fit(X_text, y_train.round().astype('int'))

predictions_text = model_text_only.predict(X_test_text)
res_text_only = classification_report(y_test.round().astype('int'), predictions_text)
print("results text_only")
print(res_text_only)
lr_text_pearsonr = stats.pearsonr(y_test, predictions_text)
print("lr_text_pearsonr: ", lr_text_pearsonr)

X_both = sparse.hstack([X_text, x_train_sims_only])
X_test_both = sparse.hstack([X_test_text, x_test_sims_only])

model_both = LogisticRegression(max_iter=1000)
model_both.fit(X_both, y_train.round().astype('int'))

predictions_both = model_both.predict(X_test_both)
res_both = classification_report(y_test.astype('int'), predictions_both)
print("Results with both similarity features and text")
print(res_both)
lr_both_pearsonr = stats.pearsonr(y_test, predictions_both)
print("lr_both_pearsonr: ", lr_both_pearsonr)

def MLP(xx, yy):
	regressor = MLPRegressor(random_state=42, max_iter=1000).fit(xx, yy)
	return regressor

regr_sims = MLP(x_train_sims_only, y_train)
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor.score
rmse_sims = regr_sims.score(x_test_sims_only, y_test)
pred_sims = regr_sims.predict(x_test_sims_only)
mlp_sims_pearsonr = stats.pearsonr(y_test, pred_sims)
print("mlp_sims_pearsonr: ", mlp_sims_pearsonr)
print("rmse_sims: ", rmse_sims)

regr_text = MLP(X_text, y_train)
rmse_text = regr_text.score(X_test_text, y_test)
pred_text = regr_text.predict(X_test_text)
mlp_sims_pearsonr = stats.pearsonr(y_test, pred_text)
print("mlp_text_pearsonr: ", mlp_sims_pearsonr)
print("rmse_text: ", rmse_text)

regr_both = MLP(X_both, y_train)
rmse_both = regr_both.score(X_test_both, y_test)
pred_both = regr_both.predict(X_test_both)
mlp_both_pearsonr = stats.pearsonr(y_test, pred_both)
print("mlp_both_pearsonr: ", mlp_sims_pearsonr)
print("rmse_both: ", rmse_both)

##############


combinations = [
				['bert_similarity'],
				['use_similarity'],
				['cos_similarity', 'bert_similarity'],
				['arccos_similarity', 'use_similarity'],
				['arccos_similarity', 'bert_similarity'],
				['arccos_similarity', 'use_similarity', 'bert_similarity'],
				['cos_similarity', 'arccos_similarity', 'use_similarity', 'bert_similarity']
			]

for comb in combinations:
    print("Running with combination: ", " ".join(comb))
    X_all = x_train[comb]
    X_test_all = x_test[comb]

    model_all = LogisticRegression(max_iter=1000)
    model_all.fit(X_all, y_train.round().astype('int'))
    predictions_all = model_all.predict(X_test_all)
    res_all = classification_report(y_test.astype('int'), predictions_all)
    print("LR classification_report")
    print(res_all)
    lr_all_pearsonr = stats.pearsonr(y_test.astype('int'), predictions_all)
    print("lr_pearsonr: ", lr_all_pearsonr)

    regr_all = MLP(X_all, y_train)
    rmse_all = regr_all.score(X_test_all, y_test)
    pred_all = regr_all.predict(X_test_all)
    mlp_all_pearsonr = stats.pearsonr(y_test, pred_all)
    print("mlp_pearsonr: ", mlp_all_pearsonr)
    print("rmse: ", rmse_all)

