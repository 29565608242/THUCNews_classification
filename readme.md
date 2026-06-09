# 1. 装依赖

pip install -r requirements.txt -i <https://pypi.tuna.tsinghua.edu.cn/simple>

# 2. 数据预处理（先确保 data/raw/ 下有 THUCNews 原始数据）

python src/data.py

# 3-5. 训练三个模型

python src/tfidf_svm.py          # 快速，几分钟
python src/bilstm.py              # 约 30 分钟（GPU）
python src/bert_model.py          # 约 1-2 小时（GPU）

# 6. 综合评估 + 可视化 + 报告

python src/evaluate.py

# 7. 数据量实验（可选）

python src/tfidf_svm.py --data_scale 5000
python src/tfidf_svm.py --data_scale 10000
