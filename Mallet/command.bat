bin\mallet import-dir --input sample-data\web --output workspace\topic-input.mallet --keep-sequence --remove-stopwords


bin\mallet import-file --input sample-data\sample\data.txt --output workspace\sample.mallet --keep-sequence --remove-stopwords

bin\mallet import-file --input sample-data\sample\label.txt --token-regex "[A-Za-z0-9]+" --output workspace\label.mallet --keep-sequence


bin\mallet import-dir --input E:\dataset\20news\20news-bydate-train --output workspace\20news_train.mallet --keep-sequence --remove-stopwords


#for making training data for 20 news
#important to use --use-pipe-from for testing data

bin\mallet import-file --input E:\dataset\20news\20_news_mallet_train\20_news_mallet_train_text.txt --output E:\dataset\20news\20_news_mallet_train\20_news_mallet_train_text.mallet --keep-sequence --remove-stopwords

bin\mallet import-file --input E:\dataset\20news\20_news_mallet_train\20_news_mallet_train_label.txt --token-regex "[A-Za-z0-9]+" --output E:\dataset\20news\20_news_mallet_train\20_news_mallet_train_label.mallet --keep-sequence

#for making testing data for 20 news

bin\mallet import-file --input E:\dataset\20news\20_news_mallet_test\20_news_mallet_test_text.txt --output E:\dataset\20news\20_news_mallet_test\20_news_mallet_test_text.mallet --keep-sequence --remove-stopwords --use-pipe-from E:\dataset\20news\20_news_mallet_train\20_news_mallet_train_text.mallet

bin\mallet import-file --input E:\dataset\20news\20_news_mallet_test\20_news_mallet_test_label.txt --token-regex "[A-Za-z0-9]+" --output E:\dataset\20news\20_news_mallet_test\20_news_mallet_test_label.mallet --keep-sequence  --use-pipe-from E:\dataset\20news\20_news_mallet_train\20_news_mallet_train_label.mallet

