python ./process/process_BEAT_bvh.py --db_path "your/source/files" \
                --save_path ./data/processed/ \
                --wavlm_model_path "your/wavlm/model/path" \
                --word2vec_model_path "your/fasttext/model/path" \
                --version v0 \
                --step step3 \
                --device cuda:0 \
                --load_type train \