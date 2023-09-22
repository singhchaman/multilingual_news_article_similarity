import os
import json
import time
from functools import reduce
import ctranslate2
import sentencepiece as spm
import pandas as pd

os.environ["CT2_VERBOSE"] = "1"

def read_json(path):
	try:
		with open(path) as fh:
			return json.load(fh)
	# Except if article was not scraped by the scraper
	except FileNotFoundError:
		return {'text': ''}

def reduce_join(df, columns):
    assert len(columns) > 1
    slist = [df[x].astype(str) for x in columns]
    return reduce(lambda x, y: x + ' ' + y, slist[1:], slist[0])

# data_dir = "data_mini"
data_dir = "data"
start = time.time()

train_records = pd.read_csv("semeval-2022_task8_train-data_batch.csv")

data_1, data_2 = [], []
# Iterate over trainng data scraped into folders and store in a csv
for pair_id in train_records["pair_id"]:
	article_1_id = pair_id.split("_")[0]
	article_2_id = pair_id.split("_")[1]
	article_1 = read_json(os.path.join(".", data_dir, article_1_id[-2:], article_1_id + ".json"))["text"].replace("\n", " ")#replace("\n\n", "\n")
	article_2 = read_json(os.path.join(".", data_dir, article_2_id[-2:], article_2_id + ".json"))["text"].replace("\n", " ")
	data_1.append(article_1)
	data_2.append(article_2)


train_records = train_records.assign(text_1=data_1)
train_records = train_records.assign(text_2=data_2)
combined = reduce_join(train_records, ["text_1", "text_2"])
train_records = pd.concat([train_records, combined], axis=1).rename(columns={0:"combined_text"})
print(train_records.head)
# train_records.to_csv("combined_data_mini.csv")
train_records.to_csv("combined_data.csv")

translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")
sp = spm.SentencePieceProcessor(model_file='./sentencepiece.model')

translated_texts = []
for lang1, lang2, text1, text2 in zip(train_records['url1_lang'], train_records['url2_lang'], train_records['text_1'], train_records['text_2']):
	if lang1 == 'de' and lang2 == 'en':
		# Encode text in sentencepiece format
		q2 = sp.encode(text2.split("\n"), out_type=str)
		text2 = translator.translate_batch(q2)
		# Decode text in sentencepiece format
		text2 = sp.decode(text2[0].hypotheses)[0]
	translated_texts.append(text2)

train_records = train_records.assign(translated_text2=translated_texts)
train_records.to_csv('translated.csv')

print("Time taken ", time.time() - start)
