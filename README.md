# vae_lyrics

### データ準備

    ./data_utanet 配下に歌詞データを格納
    ```
    $ ./create_dataset.sh
    ```
    ※ 詳細は執筆中

### 学習

    ```
    $ ./lstm_vae.sh
    ```
    または直接
    ```
    $ ./seq2seq_vae.py
    ```

### Loss や BLEU 値の確認

    Plot_graph.ipynb を起動

### 学習済みモデルから特徴ベクトルを生成

    
    ```
    $ ./extract_vector_vae.py result/
    ```

### 特徴ベクトルの主成分分析&SVMの実験

    PCA_plot.ipynb & SVM_Classifier.ipynb

### 歌詞のランダム生成 & ツイート

    ```
    $ ./bot.py
    ```
    
### 学習データから、クエリと最も類似度の高い歌詞を検索

    ```
    $ ./search_similar_lyrics.py [-h] [--result_dir RESULT_DIR] --query QUERY
    ```

### 与えた2つの歌詞から中間歌詞を生成

    ```
    $ ./interpolation.py [-h] [--result_dir RESULT_DIR] --query QUERY
    ```
    
