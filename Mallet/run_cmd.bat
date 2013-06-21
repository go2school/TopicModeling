bin\mallet.bat import-dir --input sample-data\web\en --output sample-data\input.mallet --keep-sequence --remove-stopwords

bin\mallet.bat train-topics --input sample-data\input.mallet --num-topics 10 --output-state topic-state.state --num-iterations 10 --output-model samlpe.model --output-doc-topics sample.doc_topics --output-topic-keys sample.topic_keys --num-top-words 100

bin\mallet.bat train-topics --input sample-data\input.mallet --num-topics 10 --output-state topic-state.state --num-iterations 10 --output-model samlpe.model --output-doc-topics sample.doc_topics --output-topic-keys sample.topic_keys --num-top-words 100


bin\mallet.bat hlda --input sample-data\input.mallet --output-state sample.hlda.state --num-top-words 100 --num-levels 4