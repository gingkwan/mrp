number=${1}
if [[ $number == '' ]]; then
    number=1
fi;

# Download metadata
wget --tries=100 --no-check-certificate https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_${number}.parquet 

# Download text embedding
wget --tries=100 --no-check-certificate https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/text_emb/text_emb_${number}.npy 

# Download image embedding
wget --tries=100 --no-check-certificate https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_${number}.npy 

# Apply data filtering for occupation-related captions
#python3 preprocess.py $number