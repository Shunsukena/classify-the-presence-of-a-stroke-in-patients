# classify-the-presence-of-a-stroke-in-patients
# Stroke Prediction using LightGBM

本プロジェクトでは、脳卒中（stroke）の発症を予測するために、LightGBM モデル,chatGPTを用いて分類モデルを構築しています。

## 📂 データセット概要

- カラム例：
  - `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`（目的変数）

## ⚙️ モデル概要

- 使用アルゴリズム：LightGBM
- ハイパーパラメータ：Optuna による自動チューニング済み
- クロスバリデーション：StratifiedKFold (5分割)
- スコアリング指標：F1スコア（閾値最適化）

## 🛠 前処理と特徴量エンジニアリング

LabelEncoder によるカテゴリ変数の数値変換
log変換特徴量例：
avg_glucose_level_log, bmi_log

## 🔍 評価指標
各foldごとに最適な閾値を探索し、F1スコア最大化
最終的な予測はテストデータに対して確率平均後、最適閾値で分類

## 🧪 モデル出力例
Fold 5 Best F1-score: 0.3404 (threshold=0.12)

Average Best F1-score across folds: 0.3261
Average Optimal Threshold: 0.1200

## 📌 備考
サンプリング手法（SMOTE, アンダーサンプリング）にも対応可能(精度上がらず）
