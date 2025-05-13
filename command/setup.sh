bash -c '

apt update;

apt install -y wget;

mkdir -p /workspace;

mkdir -p ~/.ssh;

chmod 700 ~/.ssh;

echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys;

chmod 700 ~/.ssh/authorized_keys;

service ssh start;

apt install python3-pip -y;

apt install git -y;

cd /root;

rm -rf ./hybrid-recommender-training;

git clone https://github.com/nudin10/hybrid-recommender-training.git;

mkdir -p ./hybrid-recommender-training/data;

ls;

wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Magazine_Subscriptions.jsonl.gz;

apt install gzip -y;

gzip -d Magazine_Subscriptions.jsonl.gz;

mv Magazine_Subscriptions.jsonl ./hybrid-recommender-training/data/Magazine_Subscriptions.jsonl;

cd hybrid-recommender-training;

apt install -y python3-venv;

apt-get install redis;

service start redis-server;

python3 -m venv venv;

source venv/bin/activate;

pip install -r requirements.txt;

sleep infinity

'