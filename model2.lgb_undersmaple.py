{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "db230789-d056-4042-9023-aea187be48f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import f1_score\n",
    "import lightgbm as lgb\n",
    "import matplotlib.pyplot as plt\n",
    "import optuna\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "from imblearn.over_sampling import SMOTENC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "65adc0bf-bb8b-4513-8218-292b0732b3e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train = pd.read_csv(\"/Users/natani/Desktop/Python/コンペ/kaggle spring 2025/train.csv\", index_col=0)\n",
    "df_test = pd.read_csv(\"/Users/natani/Desktop/Python/コンペ/kaggle spring 2025/test.csv\", index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "f99c2dbb-b2bc-428a-b9ca-664bdc3cf1bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 12243 entries, 1 to 12243\n",
      "Data columns (total 18 columns):\n",
      " #   Column                 Non-Null Count  Dtype  \n",
      "---  ------                 --------------  -----  \n",
      " 0   gender                 12243 non-null  int64  \n",
      " 1   age                    12243 non-null  float64\n",
      " 2   hypertension           12243 non-null  int64  \n",
      " 3   heart_disease          12243 non-null  int64  \n",
      " 4   ever_married           12243 non-null  int64  \n",
      " 5   work_type              12243 non-null  int64  \n",
      " 6   Residence_type         12243 non-null  int64  \n",
      " 7   avg_glucose_level      12243 non-null  float64\n",
      " 8   bmi                    12243 non-null  float64\n",
      " 9   smoking_status         12243 non-null  int64  \n",
      " 10  stroke                 12243 non-null  int64  \n",
      " 11  avg_glucose_level_log  12243 non-null  float64\n",
      " 12  bmi_log                12243 non-null  float64\n",
      " 13  age*bmi                12243 non-null  float64\n",
      " 14  glucose*bmi            12243 non-null  float64\n",
      " 15  heart+age              12243 non-null  float64\n",
      " 16  hypertension+age       12243 non-null  float64\n",
      " 17  work+residence         12243 non-null  int64  \n",
      "dtypes: float64(9), int64(9)\n",
      "memory usage: 1.8 MB\n"
     ]
    }
   ],
   "source": [
    "df_train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "47be132b-c22a-4921-9ddb-7b85d0618927",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== gender ===\n",
      "gender\n",
      "Female    7563\n",
      "Male      4679\n",
      "Other        1\n",
      "Name: count, dtype: int64\n",
      "\n",
      "\n",
      "=== age ===\n",
      "age\n",
      "57.00    272\n",
      "78.00    268\n",
      "53.00    268\n",
      "31.00    246\n",
      "45.00    246\n",
      "        ... \n",
      "0.24       5\n",
      "0.40       5\n",
      "0.48       3\n",
      "1.30       2\n",
      "0.68       1\n",
      "Name: count, Length: 106, dtype: int64\n",
      "\n",
      "\n",
      "=== hypertension ===\n",
      "hypertension\n",
      "0    11638\n",
      "1      605\n",
      "Name: count, dtype: int64\n",
      "\n",
      "\n",
      "=== heart_disease ===\n",
      "heart_disease\n",
      "0    11958\n",
      "1      285\n",
      "Name: count, dtype: int64\n",
      "\n",
      "\n",
      "=== ever_married ===\n",
      "ever_married\n",
      "Yes    8275\n",
      "No     3968\n",
      "Name: count, dtype: int64\n",
      "\n",
      "\n",
      "=== work_type ===\n",
      "work_type\n",
      "Private          7791\n",
      "children         1667\n",
      "Self-employed    1539\n",
      "Govt_job         1215\n",
      "Never_worked       31\n",
      "Name: count, dtype: int64\n",
      "\n",
      "\n",
      "=== Residence_type ===\n",
      "Residence_type\n",
      "Urban    6124\n",
      "Rural    6119\n",
      "Name: count, dtype: int64\n",
      "\n",
      "\n",
      "=== avg_glucose_level ===\n",
      "avg_glucose_level\n",
      "77.55     29\n",
      "85.84     27\n",
      "91.85     26\n",
      "93.88     25\n",
      "84.03     25\n",
      "          ..\n",
      "87.11      1\n",
      "87.66      1\n",
      "114.16     1\n",
      "75.60      1\n",
      "93.11      1\n",
      "Name: count, Length: 3449, dtype: int64\n",
      "\n",
      "\n",
      "=== bmi ===\n",
      "bmi\n",
      "28.70    136\n",
      "23.40    134\n",
      "28.40    127\n",
      "26.70    120\n",
      "26.10    113\n",
      "        ... \n",
      "54.80      1\n",
      "56.60      1\n",
      "47.20      1\n",
      "53.80      1\n",
      "20.22      1\n",
      "Name: count, Length: 395, dtype: int64\n",
      "\n",
      "\n",
      "=== smoking_status ===\n",
      "smoking_status\n",
      "never smoked       5029\n",
      "Unknown            3651\n",
      "formerly smoked    1839\n",
      "smokes             1724\n",
      "Name: count, dtype: int64\n",
      "\n",
      "\n",
      "=== stroke ===\n",
      "stroke\n",
      "0    11737\n",
      "1      506\n",
      "Name: count, dtype: int64\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "for col in df_train.columns:\n",
    "    print(f\"=== {col} ===\")\n",
    "    print(df_train[col].value_counts())\n",
    "    print(\"\\n\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "96937281-78e6-4311-9396-c20b76921b3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def datapre(df):\n",
    "    df['avg_glucose_level_log'] = np.log1p(df['avg_glucose_level'])\n",
    "    df['bmi_log'] = np.log1p(df['bmi'])\n",
    "    df[\"age*bmi\"] = df[\"age\"]*df[\"bmi\"]\n",
    "    df[\"glucose*bmi\"]= df[\"avg_glucose_level\"]*df[\"bmi\"]\n",
    "    df[\"heart+age\"] = df[\"heart_disease\"]+df[\"age\"]\n",
    "    df[\"hypertension+age\"] = df[\"hypertension\"] + df[\"age\"]\n",
    "    df[\"work+residence\"] = df[\"work_type\"]+df[\"Residence_type\"]\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "7263fb3d-7a32-4f02-b5ca-1c327498c4f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train = datapre(df_train)\n",
    "df_test = datapre(df_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "25ab9078-8e95-4b70-8c9b-464c8352654a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.15653\n",
      "Fold 1 Best F1-score: 0.3300 (threshold=0.12)\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.157543\n",
      "Fold 2 Best F1-score: 0.3043 (threshold=0.12)\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.158519\n",
      "Fold 3 Best F1-score: 0.3066 (threshold=0.11)\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.154084\n",
      "Fold 4 Best F1-score: 0.3494 (threshold=0.13)\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.1568\n",
      "Fold 5 Best F1-score: 0.3404 (threshold=0.12)\n",
      "\n",
      "Average Best F1-score across folds: 0.3261\n",
      "Average Optimal Threshold: 0.1200\n"
     ]
    }
   ],
   "source": [
    "# カテゴリ変数のエンコーディング\n",
    "cat_cols = df_train.select_dtypes(include=\"object\").columns\n",
    "le_dict = {}\n",
    "for col in cat_cols:\n",
    "    le = LabelEncoder()\n",
    "    df_train[col] = le.fit_transform(df_train[col])\n",
    "    df_test[col] = le.transform(df_test[col])\n",
    "    le_dict[col] = le\n",
    "\n",
    "# 最適なハイパーパラメータ（Optunaでチューニング後の結果）\n",
    "best_params = {\n",
    "    'learning_rate': 0.006927918069998695, 'num_leaves': 83, 'max_depth': 4, 'min_child_samples': 47, 'subsample': 0.6160943288395566, 'colsample_bytree': 0.5179983582294293, 'reg_alpha': 0.0014196850444548798, 'reg_lambda': 0.00027946017747999287\n",
    "}\n",
    "\n",
    "# 特徴量と目的変数\n",
    "X = df_train.drop(\"stroke\", axis=1)\n",
    "y = df_train[\"stroke\"]\n",
    "X_test = df_test.copy()\n",
    "\n",
    "# サンプリング手法を選択: \"over\"（SMOTE） or \"under\"（ランダムアンダーサンプリング） or None\n",
    "sampling_method = None  # \"under\", \"over\", or None\n",
    "\n",
    "# StratifiedKFold 5分割\n",
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "f1_scores = []\n",
    "thresholds_list = []\n",
    "test_preds = np.zeros(len(X_test))\n",
    "\n",
    "for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):\n",
    "    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]\n",
    "    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]\n",
    "\n",
    "    # サンプリングの適用（trainのみに！）\n",
    "    if sampling_method == \"over\":\n",
    "        smote = SMOTE(random_state=42)\n",
    "        X_train, y_train = smote.fit_resample(X_train, y_train)\n",
    "    elif sampling_method == \"under\":\n",
    "        rus = RandomUnderSampler(random_state=42)\n",
    "        X_train, y_train = rus.fit_resample(X_train, y_train)\n",
    "\n",
    "    # LGBMモデル定義\n",
    "    model = lgb.LGBMClassifier(\n",
    "        objective=\"binary\",\n",
    "        is_unbalance=(sampling_method is None),  # サンプリング時はFalseでOK\n",
    "        random_state=42,\n",
    "        n_estimators=1000,\n",
    "        **best_params,\n",
    "        verbose=-1\n",
    "    )\n",
    "\n",
    "    model.fit(\n",
    "        X_train, y_train,\n",
    "        eval_set=[(X_val, y_val)],\n",
    "        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]\n",
    "    )\n",
    "\n",
    "    y_prob = model.predict_proba(X_val)[:, 1]\n",
    "\n",
    "    # 閾値ごとのF1スコア\n",
    "    thresholds = np.arange(0.01, 1.00, 0.01)\n",
    "    f1s = [f1_score(y_val, (y_prob > t).astype(int), zero_division=0) for t in thresholds]\n",
    "    best_threshold = thresholds[np.argmax(f1s)]\n",
    "    best_f1 = max(f1s)\n",
    "    f1_scores.append(best_f1)\n",
    "    thresholds_list.append(best_threshold)\n",
    "\n",
    "    print(f\"Fold {fold+1} Best F1-score: {best_f1:.4f} (threshold={best_threshold:.2f})\")\n",
    "\n",
    "    # test予測（確率）\n",
    "    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits\n",
    "\n",
    "# 平均の最適閾値\n",
    "avg_best_threshold = np.mean(thresholds_list)\n",
    "print(f\"\\nAverage Best F1-score across folds: {np.mean(f1_scores):.4f}\")\n",
    "print(f\"Average Optimal Threshold: {avg_best_threshold:.4f}\")\n",
    "\n",
    "# 最終予測（0 or 1）に変換\n",
    "final_preds = (test_preds > avg_best_threshold).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "ff2e06a8-8658-48cf-8d4b-a8d0ddfc33d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 提出用データフレーム作成\n",
    "submission = pd.DataFrame({\n",
    "    \"id\": df_test.index,\n",
    "    \"stroke\": final_preds\n",
    "})\n",
    "\n",
    "# 保存\n",
    "submission.to_csv(\"submission.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "6755b840-81f0-4798-b3c8-0388e8e9e234",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:41,874] A new study created in memory with name: no-name-306ac49a-eedb-48c6-9b4f-4b0f7258d65e\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[1]\tvalid_0's binary_logloss: 0.155896\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[4]\tvalid_0's binary_logloss: 0.157515\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[1]\tvalid_0's binary_logloss: 0.157546\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[4]\tvalid_0's binary_logloss: 0.151493\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[1]\tvalid_0's binary_logloss: 0.155207\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:43,781] Trial 0 finished with value: 0.30241164028963985 and parameters: {'learning_rate': 0.04731910011028515, 'num_leaves': 142, 'max_depth': 13, 'min_child_samples': 38, 'subsample': 0.8008689469926504, 'colsample_bytree': 0.5098625402181186, 'reg_alpha': 0.004471443052551391, 'reg_lambda': 0.0091398426654955}. Best is trial 0 with value: 0.30241164028963985.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[29]\tvalid_0's binary_logloss: 0.147194\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[32]\tvalid_0's binary_logloss: 0.150163\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[30]\tvalid_0's binary_logloss: 0.152166\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[37]\tvalid_0's binary_logloss: 0.143404\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[32]\tvalid_0's binary_logloss: 0.14944\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:46,024] Trial 1 finished with value: 0.3187791831651754 and parameters: {'learning_rate': 0.007603128684843798, 'num_leaves': 121, 'max_depth': 13, 'min_child_samples': 32, 'subsample': 0.5355293984439232, 'colsample_bytree': 0.8717018202959117, 'reg_alpha': 0.00040961689434883356, 'reg_lambda': 2.8792311472672905e-05}. Best is trial 1 with value: 0.3187791831651754.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.151434\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[5]\tvalid_0's binary_logloss: 0.152746\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[5]\tvalid_0's binary_logloss: 0.153524\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.148247\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[5]\tvalid_0's binary_logloss: 0.153179\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:47,757] Trial 2 finished with value: 0.31475889837642496 and parameters: {'learning_rate': 0.02665072096971644, 'num_leaves': 82, 'max_depth': 9, 'min_child_samples': 46, 'subsample': 0.6438165854666325, 'colsample_bytree': 0.744995650288209, 'reg_alpha': 0.012648173465048379, 'reg_lambda': 7.12727821040019e-05}. Best is trial 1 with value: 0.3187791831651754.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[1]\tvalid_0's binary_logloss: 0.157729\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[1]\tvalid_0's binary_logloss: 0.159213\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[1]\tvalid_0's binary_logloss: 0.159775\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[1]\tvalid_0's binary_logloss: 0.157904\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[1]\tvalid_0's binary_logloss: 0.158114\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:49,307] Trial 3 finished with value: 0.31151128947154805 and parameters: {'learning_rate': 0.03671870783191966, 'num_leaves': 145, 'max_depth': 4, 'min_child_samples': 19, 'subsample': 0.9110662084694463, 'colsample_bytree': 0.5420827918496367, 'reg_alpha': 2.6842975549356684e-05, 'reg_lambda': 0.0006239289526460897}. Best is trial 1 with value: 0.3187791831651754.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.14472\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[21]\tvalid_0's binary_logloss: 0.15055\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[23]\tvalid_0's binary_logloss: 0.150507\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[24]\tvalid_0's binary_logloss: 0.141217\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[18]\tvalid_0's binary_logloss: 0.148125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:51,488] Trial 4 finished with value: 0.3075948505829067 and parameters: {'learning_rate': 0.012032578410336229, 'num_leaves': 97, 'max_depth': 13, 'min_child_samples': 10, 'subsample': 0.6423794635730288, 'colsample_bytree': 0.9996086285825567, 'reg_alpha': 0.0006357043736240401, 'reg_lambda': 0.00059347514346965}. Best is trial 1 with value: 0.3187791831651754.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[18]\tvalid_0's binary_logloss: 0.15202\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[21]\tvalid_0's binary_logloss: 0.153763\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.154344\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[22]\tvalid_0's binary_logloss: 0.149012\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.152005\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:53,273] Trial 5 finished with value: 0.32033232679394325 and parameters: {'learning_rate': 0.007685745811883094, 'num_leaves': 35, 'max_depth': 19, 'min_child_samples': 27, 'subsample': 0.7838686596618676, 'colsample_bytree': 0.7194233625705334, 'reg_alpha': 0.06249517897052847, 'reg_lambda': 2.064163672782636e-05}. Best is trial 5 with value: 0.32033232679394325.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.150056\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[5]\tvalid_0's binary_logloss: 0.151888\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[5]\tvalid_0's binary_logloss: 0.152435\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.146839\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[4]\tvalid_0's binary_logloss: 0.151541\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:55,200] Trial 6 finished with value: 0.3180289742210996 and parameters: {'learning_rate': 0.03135257546789046, 'num_leaves': 245, 'max_depth': 12, 'min_child_samples': 45, 'subsample': 0.5730240235844852, 'colsample_bytree': 0.8072156290649655, 'reg_alpha': 4.617041018366946e-05, 'reg_lambda': 0.005109504909307646}. Best is trial 5 with value: 0.32033232679394325.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[5]\tvalid_0's binary_logloss: 0.148294\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[5]\tvalid_0's binary_logloss: 0.150916\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[5]\tvalid_0's binary_logloss: 0.154127\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.145417\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[4]\tvalid_0's binary_logloss: 0.150897\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:57,093] Trial 7 finished with value: 0.3143138741387833 and parameters: {'learning_rate': 0.03639834895282528, 'num_leaves': 288, 'max_depth': 15, 'min_child_samples': 40, 'subsample': 0.7372802930835464, 'colsample_bytree': 0.9396093196575908, 'reg_alpha': 0.0007714283819023173, 'reg_lambda': 0.004127119689823113}. Best is trial 5 with value: 0.32033232679394325.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.153175\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.154962\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.156568\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.1513\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[14]\tvalid_0's binary_logloss: 0.154388\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:49:59,194] Trial 8 finished with value: 0.32399697725581156 and parameters: {'learning_rate': 0.0100332121059587, 'num_leaves': 30, 'max_depth': 6, 'min_child_samples': 30, 'subsample': 0.5469156999730496, 'colsample_bytree': 0.7500452584991719, 'reg_alpha': 0.0987521224053487, 'reg_lambda': 0.00044266203392696673}. Best is trial 8 with value: 0.32399697725581156.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[3]\tvalid_0's binary_logloss: 0.152444\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[3]\tvalid_0's binary_logloss: 0.152956\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[2]\tvalid_0's binary_logloss: 0.154174\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[3]\tvalid_0's binary_logloss: 0.146775\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[3]\tvalid_0's binary_logloss: 0.152489\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:01,107] Trial 9 finished with value: 0.3125907512551325 and parameters: {'learning_rate': 0.04967410057133842, 'num_leaves': 94, 'max_depth': 18, 'min_child_samples': 46, 'subsample': 0.5045386559272009, 'colsample_bytree': 0.6860656224880624, 'reg_alpha': 0.0007357940071786507, 'reg_lambda': 0.0009917571163503248}. Best is trial 8 with value: 0.32399697725581156.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.156174\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.157342\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.157925\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.153859\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.156212\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:02,675] Trial 10 finished with value: 0.32708850968339576 and parameters: {'learning_rate': 0.01596141563682791, 'num_leaves': 23, 'max_depth': 3, 'min_child_samples': 22, 'subsample': 0.9957762847708345, 'colsample_bytree': 0.6254573782636885, 'reg_alpha': 0.07856604344540212, 'reg_lambda': 0.08998106870282399}. Best is trial 10 with value: 0.32708850968339576.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.156164\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.157348\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.158154\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.153961\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.156196\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:04,215] Trial 11 finished with value: 0.31986434257096774 and parameters: {'learning_rate': 0.01641547584466329, 'num_leaves': 29, 'max_depth': 3, 'min_child_samples': 22, 'subsample': 0.9933037907517202, 'colsample_bytree': 0.6275613718636761, 'reg_alpha': 0.0798818615864222, 'reg_lambda': 0.0001675330431973544}. Best is trial 10 with value: 0.32708850968339576.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.153914\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.155458\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.156181\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.151349\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.153851\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:05,844] Trial 12 finished with value: 0.31879596632590196 and parameters: {'learning_rate': 0.014090611835223567, 'num_leaves': 22, 'max_depth': 7, 'min_child_samples': 12, 'subsample': 0.8991321073212037, 'colsample_bytree': 0.6399880772658073, 'reg_alpha': 0.014635626210045863, 'reg_lambda': 0.09923500111636191}. Best is trial 10 with value: 0.32708850968339576.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[25]\tvalid_0's binary_logloss: 0.152289\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[25]\tvalid_0's binary_logloss: 0.154731\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[25]\tvalid_0's binary_logloss: 0.156113\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[30]\tvalid_0's binary_logloss: 0.150846\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[26]\tvalid_0's binary_logloss: 0.153861\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:07,595] Trial 13 finished with value: 0.3274885206968058 and parameters: {'learning_rate': 0.00526737434596951, 'num_leaves': 193, 'max_depth': 6, 'min_child_samples': 30, 'subsample': 0.6984358400995776, 'colsample_bytree': 0.8092947625895537, 'reg_alpha': 0.024826014391231426, 'reg_lambda': 0.07372900772602071}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[40]\tvalid_0's binary_logloss: 0.148173\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[35]\tvalid_0's binary_logloss: 0.153254\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[32]\tvalid_0's binary_logloss: 0.153254\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[46]\tvalid_0's binary_logloss: 0.146237\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[39]\tvalid_0's binary_logloss: 0.152308\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:09,680] Trial 14 finished with value: 0.3086503828108559 and parameters: {'learning_rate': 0.005129567709485328, 'num_leaves': 200, 'max_depth': 9, 'min_child_samples': 18, 'subsample': 0.7020414223859357, 'colsample_bytree': 0.8552577950544613, 'reg_alpha': 0.019596919104010645, 'reg_lambda': 0.09429919128104979}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.156939\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.15804\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.15911\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.15468\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.157431\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:11,294] Trial 15 finished with value: 0.31861882517354145 and parameters: {'learning_rate': 0.018515945712767923, 'num_leaves': 191, 'max_depth': 5, 'min_child_samples': 25, 'subsample': 0.8575769444357368, 'colsample_bytree': 0.5737418497146887, 'reg_alpha': 0.0028168626955191035, 'reg_lambda': 0.024365472868659185}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.149936\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.153612\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.154436\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.148006\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.15345\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:13,027] Trial 16 finished with value: 0.3250334451978175 and parameters: {'learning_rate': 0.021406103362239783, 'num_leaves': 190, 'max_depth': 8, 'min_child_samples': 35, 'subsample': 0.9903722337838529, 'colsample_bytree': 0.7985562546951096, 'reg_alpha': 0.0001116133844733616, 'reg_lambda': 0.02642361744710451}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.156254\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.157283\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.158018\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[17]\tvalid_0's binary_logloss: 0.154005\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.156489\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:14,608] Trial 17 finished with value: 0.32254807481209047 and parameters: {'learning_rate': 0.00662628165014151, 'num_leaves': 231, 'max_depth': 3, 'min_child_samples': 16, 'subsample': 0.6825169083944942, 'colsample_bytree': 0.6319767420279809, 'reg_alpha': 0.029593610914949567, 'reg_lambda': 0.02642975987949004}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[18]\tvalid_0's binary_logloss: 0.148385\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.153404\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[17]\tvalid_0's binary_logloss: 0.152163\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[21]\tvalid_0's binary_logloss: 0.145273\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.150282\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:16,538] Trial 18 finished with value: 0.31108658500010583 and parameters: {'learning_rate': 0.011128354173564892, 'num_leaves': 64, 'max_depth': 10, 'min_child_samples': 5, 'subsample': 0.8334474424089277, 'colsample_bytree': 0.6757595621082305, 'reg_alpha': 0.0043486725374955295, 'reg_lambda': 0.0030334728043121635}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.152003\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.155099\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.155924\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.151125\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.154233\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:18,203] Trial 19 finished with value: 0.3183419687336581 and parameters: {'learning_rate': 0.020806164430235887, 'num_leaves': 283, 'max_depth': 6, 'min_child_samples': 24, 'subsample': 0.9221309272980991, 'colsample_bytree': 0.9056676053062584, 'reg_alpha': 0.006349275160199501, 'reg_lambda': 0.056099721536265613}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[24]\tvalid_0's binary_logloss: 0.154028\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[24]\tvalid_0's binary_logloss: 0.155723\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[24]\tvalid_0's binary_logloss: 0.156729\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[27]\tvalid_0's binary_logloss: 0.152093\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[25]\tvalid_0's binary_logloss: 0.154781\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:19,883] Trial 20 finished with value: 0.32056685276022145 and parameters: {'learning_rate': 0.005219264063080544, 'num_leaves': 229, 'max_depth': 5, 'min_child_samples': 30, 'subsample': 0.6128358817099302, 'colsample_bytree': 0.7879377719178977, 'reg_alpha': 0.03528759158401972, 'reg_lambda': 0.011601767590104882}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.150915\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.154271\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.155204\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.149988\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.152996\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:21,621] Trial 21 finished with value: 0.3229606736778005 and parameters: {'learning_rate': 0.02045827087986109, 'num_leaves': 181, 'max_depth': 7, 'min_child_samples': 35, 'subsample': 0.9997453100517187, 'colsample_bytree': 0.8058669095450065, 'reg_alpha': 0.00012331885827740854, 'reg_lambda': 0.03163471504591885}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[12]\tvalid_0's binary_logloss: 0.150106\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.153501\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.154218\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.148614\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[12]\tvalid_0's binary_logloss: 0.153117\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:23,516] Trial 22 finished with value: 0.323803122385431 and parameters: {'learning_rate': 0.014041746129586457, 'num_leaves': 177, 'max_depth': 8, 'min_child_samples': 35, 'subsample': 0.958677741336755, 'colsample_bytree': 0.8430212737697713, 'reg_alpha': 0.00023346408458550224, 'reg_lambda': 0.03572484304030881}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.150777\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.15232\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.153127\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.146385\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.152526\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:25,468] Trial 23 finished with value: 0.3199223767411639 and parameters: {'learning_rate': 0.023172982813283147, 'num_leaves': 209, 'max_depth': 10, 'min_child_samples': 42, 'subsample': 0.9508075898928009, 'colsample_bytree': 0.7775291317101926, 'reg_alpha': 0.0016985742517573287, 'reg_lambda': 0.011569689369756853}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.154897\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.156632\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.157152\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.152789\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[6]\tvalid_0's binary_logloss: 0.155172\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:27,047] Trial 24 finished with value: 0.32521985336315196 and parameters: {'learning_rate': 0.016665116987413423, 'num_leaves': 162, 'max_depth': 4, 'min_child_samples': 34, 'subsample': 0.8655692117220248, 'colsample_bytree': 0.7051817598321315, 'reg_alpha': 1.3622102023783777e-05, 'reg_lambda': 0.06938637050704695}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.157661\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.158545\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.159102\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[12]\tvalid_0's binary_logloss: 0.155331\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.157709\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:28,622] Trial 25 finished with value: 0.32062090523404024 and parameters: {'learning_rate': 0.009647973595884764, 'num_leaves': 150, 'max_depth': 3, 'min_child_samples': 21, 'subsample': 0.8664398427429159, 'colsample_bytree': 0.5833582655106555, 'reg_alpha': 1.740141185505233e-05, 'reg_lambda': 0.06091053728570127}. Best is trial 13 with value: 0.3274885206968058.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.154172\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.15595\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.156844\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.152373\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[7]\tvalid_0's binary_logloss: 0.154992\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:30,256] Trial 26 finished with value: 0.32751771165226523 and parameters: {'learning_rate': 0.016382137792361216, 'num_leaves': 124, 'max_depth': 5, 'min_child_samples': 50, 'subsample': 0.7496813320095204, 'colsample_bytree': 0.6910003968921627, 'reg_alpha': 0.0441399843992919, 'reg_lambda': 0.013852702580530554}. Best is trial 26 with value: 0.32751771165226523.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.154203\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.156026\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.157004\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.152105\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[8]\tvalid_0's binary_logloss: 0.154823\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:31,905] Trial 27 finished with value: 0.3190565299696591 and parameters: {'learning_rate': 0.01376143500862835, 'num_leaves': 64, 'max_depth': 5, 'min_child_samples': 50, 'subsample': 0.7524530958093242, 'colsample_bytree': 0.6588465349039431, 'reg_alpha': 0.03229965650811086, 'reg_lambda': 0.0015906114785591825}. Best is trial 26 with value: 0.32751771165226523.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.154884\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[14]\tvalid_0's binary_logloss: 0.156239\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[14]\tvalid_0's binary_logloss: 0.157679\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[17]\tvalid_0's binary_logloss: 0.152113\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[14]\tvalid_0's binary_logloss: 0.15563\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:33,747] Trial 28 finished with value: 0.3212298038886606 and parameters: {'learning_rate': 0.00895495159329047, 'num_leaves': 125, 'max_depth': 6, 'min_child_samples': 50, 'subsample': 0.7107108619018786, 'colsample_bytree': 0.5942552286841645, 'reg_alpha': 0.009781932324065925, 'reg_lambda': 0.00936770173427838}. Best is trial 26 with value: 0.32751771165226523.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.14463\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.149999\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[12]\tvalid_0's binary_logloss: 0.147774\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.139098\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[14]\tvalid_0's binary_logloss: 0.148259\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:36,303] Trial 29 finished with value: 0.32434293495318955 and parameters: {'learning_rate': 0.02606425943560396, 'num_leaves': 120, 'max_depth': 16, 'min_child_samples': 14, 'subsample': 0.7839944655168353, 'colsample_bytree': 0.738627095079339, 'reg_alpha': 0.047689392863397025, 'reg_lambda': 0.014079373698513274}. Best is trial 26 with value: 0.32751771165226523.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.157004\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.158147\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.15908\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.154839\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.157409\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:37,976] Trial 30 finished with value: 0.32761338696091286 and parameters: {'learning_rate': 0.01166335137453747, 'num_leaves': 54, 'max_depth': 4, 'min_child_samples': 39, 'subsample': 0.815799498499871, 'colsample_bytree': 0.5065435917589096, 'reg_alpha': 0.007443467696254963, 'reg_lambda': 0.043816184051142945}. Best is trial 30 with value: 0.32761338696091286.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.157002\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.158142\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.159074\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.154826\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.157403\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:39,600] Trial 31 finished with value: 0.3285928010924839 and parameters: {'learning_rate': 0.011634077138078852, 'num_leaves': 52, 'max_depth': 4, 'min_child_samples': 39, 'subsample': 0.8238595403817219, 'colsample_bytree': 0.5153997626502692, 'reg_alpha': 0.01778192003805539, 'reg_lambda': 0.04448417842912388}. Best is trial 31 with value: 0.3285928010924839.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[18]\tvalid_0's binary_logloss: 0.15622\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.157422\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[18]\tvalid_0's binary_logloss: 0.158452\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.153963\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.156739\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:41,243] Trial 32 finished with value: 0.3239965845738765 and parameters: {'learning_rate': 0.006213636705164127, 'num_leaves': 64, 'max_depth': 4, 'min_child_samples': 39, 'subsample': 0.8166127333475964, 'colsample_bytree': 0.502819600470312, 'reg_alpha': 0.006702499828611332, 'reg_lambda': 0.01717483916973988}. Best is trial 31 with value: 0.3285928010924839.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.153164\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.155249\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[17]\tvalid_0's binary_logloss: 0.157422\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.151082\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[17]\tvalid_0's binary_logloss: 0.154891\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:43,067] Trial 33 finished with value: 0.32109288362995214 and parameters: {'learning_rate': 0.00812263106341505, 'num_leaves': 56, 'max_depth': 7, 'min_child_samples': 42, 'subsample': 0.7698143201957987, 'colsample_bytree': 0.5300842896824951, 'reg_alpha': 0.01879462565031499, 'reg_lambda': 0.04828540111842113}. Best is trial 31 with value: 0.3285928010924839.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.156294\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.157643\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.158906\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.153794\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.15737\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:44,716] Trial 34 finished with value: 0.3248035810873954 and parameters: {'learning_rate': 0.0122324836145533, 'num_leaves': 113, 'max_depth': 5, 'min_child_samples': 42, 'subsample': 0.736349990414843, 'colsample_bytree': 0.5413164219264077, 'reg_alpha': 0.00262106248428899, 'reg_lambda': 0.00671304244476628}. Best is trial 31 with value: 0.3285928010924839.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.156326\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[17]\tvalid_0's binary_logloss: 0.157527\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.158508\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.154032\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[17]\tvalid_0's binary_logloss: 0.156805\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:46,335] Trial 35 finished with value: 0.32957403318794526 and parameters: {'learning_rate': 0.0067297389384294755, 'num_leaves': 85, 'max_depth': 4, 'min_child_samples': 38, 'subsample': 0.6556470742829449, 'colsample_bytree': 0.5421951367666594, 'reg_alpha': 0.00861468679313144, 'reg_lambda': 0.018015845480501544}. Best is trial 35 with value: 0.32957403318794526.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.15685\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.158111\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.158596\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.154527\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.157249\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:47,940] Trial 36 finished with value: 0.3286924915201973 and parameters: {'learning_rate': 0.011295406766791886, 'num_leaves': 94, 'max_depth': 4, 'min_child_samples': 48, 'subsample': 0.6399942924181722, 'colsample_bytree': 0.562716102422409, 'reg_alpha': 0.008185590870101147, 'reg_lambda': 0.002331386734344951}. Best is trial 35 with value: 0.32957403318794526.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.156955\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.15802\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[9]\tvalid_0's binary_logloss: 0.158984\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.154764\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.157453\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:49,537] Trial 37 finished with value: 0.3245131257627063 and parameters: {'learning_rate': 0.011039616976716056, 'num_leaves': 48, 'max_depth': 4, 'min_child_samples': 38, 'subsample': 0.6480046354992348, 'colsample_bytree': 0.555265755697794, 'reg_alpha': 0.009149295087417542, 'reg_lambda': 0.002358184347229369}. Best is trial 35 with value: 0.32957403318794526.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.1565\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.157514\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.158532\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.15403\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.156747\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:51,178] Trial 38 finished with value: 0.3300425481836601 and parameters: {'learning_rate': 0.006927918069998695, 'num_leaves': 83, 'max_depth': 4, 'min_child_samples': 47, 'subsample': 0.6160943288395566, 'colsample_bytree': 0.5179983582294293, 'reg_alpha': 0.0014196850444548798, 'reg_lambda': 0.00027946017747999287}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[31]\tvalid_0's binary_logloss: 0.150575\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[28]\tvalid_0's binary_logloss: 0.152753\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[26]\tvalid_0's binary_logloss: 0.154008\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[35]\tvalid_0's binary_logloss: 0.147033\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[28]\tvalid_0's binary_logloss: 0.15203\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:53,221] Trial 39 finished with value: 0.3268746157309236 and parameters: {'learning_rate': 0.006439170460463075, 'num_leaves': 85, 'max_depth': 11, 'min_child_samples': 46, 'subsample': 0.600318027463129, 'colsample_bytree': 0.5981196466973658, 'reg_alpha': 0.0015971771116452052, 'reg_lambda': 0.00022327783062371127}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[27]\tvalid_0's binary_logloss: 0.149868\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[24]\tvalid_0's binary_logloss: 0.152673\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[24]\tvalid_0's binary_logloss: 0.154064\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[29]\tvalid_0's binary_logloss: 0.14674\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[27]\tvalid_0's binary_logloss: 0.151613\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:55,321] Trial 40 finished with value: 0.32651793137938173 and parameters: {'learning_rate': 0.007249833518671733, 'num_leaves': 79, 'max_depth': 15, 'min_child_samples': 48, 'subsample': 0.6612970534969614, 'colsample_bytree': 0.5298043509079057, 'reg_alpha': 0.0004788932583041413, 'reg_lambda': 3.697037612822749e-05}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.156616\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.15771\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[12]\tvalid_0's binary_logloss: 0.15876\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.15429\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.157075\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:56,973] Trial 41 finished with value: 0.3282699110213488 and parameters: {'learning_rate': 0.00863665050035242, 'num_leaves': 104, 'max_depth': 4, 'min_child_samples': 44, 'subsample': 0.6104191081802646, 'colsample_bytree': 0.5032302388712698, 'reg_alpha': 0.0051979674710682855, 'reg_lambda': 0.00026840924409041796}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.157468\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.158387\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[12]\tvalid_0's binary_logloss: 0.159007\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[14]\tvalid_0's binary_logloss: 0.155141\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[13]\tvalid_0's binary_logloss: 0.157544\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:50:58,588] Trial 42 finished with value: 0.3214984800762346 and parameters: {'learning_rate': 0.008439469994898422, 'num_leaves': 107, 'max_depth': 3, 'min_child_samples': 44, 'subsample': 0.5879875763421871, 'colsample_bytree': 0.5668764646160046, 'reg_alpha': 0.0037070318478484026, 'reg_lambda': 0.00010910768623725587}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.156458\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.157495\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[15]\tvalid_0's binary_logloss: 0.158653\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[18]\tvalid_0's binary_logloss: 0.154166\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[16]\tvalid_0's binary_logloss: 0.156809\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:51:00,223] Trial 43 finished with value: 0.32854956143293135 and parameters: {'learning_rate': 0.007294958725804471, 'num_leaves': 80, 'max_depth': 4, 'min_child_samples': 44, 'subsample': 0.6155835045070385, 'colsample_bytree': 0.5220963658442946, 'reg_alpha': 0.0012786443148623938, 'reg_lambda': 0.00025440892942606304}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[22]\tvalid_0's binary_logloss: 0.152589\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[21]\tvalid_0's binary_logloss: 0.154989\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.156315\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[25]\tvalid_0's binary_logloss: 0.149709\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.15434\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:51:02,108] Trial 44 finished with value: 0.3267868021384388 and parameters: {'learning_rate': 0.00728765720986787, 'num_leaves': 80, 'max_depth': 8, 'min_child_samples': 48, 'subsample': 0.5548850962271943, 'colsample_bytree': 0.5326711267735942, 'reg_alpha': 0.0012136802865027453, 'reg_lambda': 0.0005601885561106459}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[22]\tvalid_0's binary_logloss: 0.153865\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[22]\tvalid_0's binary_logloss: 0.155935\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[21]\tvalid_0's binary_logloss: 0.157637\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[26]\tvalid_0's binary_logloss: 0.151806\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[21]\tvalid_0's binary_logloss: 0.155532\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:51:03,912] Trial 45 finished with value: 0.3248755985607993 and parameters: {'learning_rate': 0.005850339850883331, 'num_leaves': 39, 'max_depth': 6, 'min_child_samples': 37, 'subsample': 0.6220890484726349, 'colsample_bytree': 0.5575665747423268, 'reg_alpha': 0.0024140637498100224, 'reg_lambda': 0.0010599179614205335}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.157663\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.158605\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[10]\tvalid_0's binary_logloss: 0.159209\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[12]\tvalid_0's binary_logloss: 0.155267\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[11]\tvalid_0's binary_logloss: 0.157733\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:51:05,530] Trial 46 finished with value: 0.3228332038508103 and parameters: {'learning_rate': 0.009841280435969596, 'num_leaves': 89, 'max_depth': 3, 'min_child_samples': 41, 'subsample': 0.5272123444324134, 'colsample_bytree': 0.605385767870251, 'reg_alpha': 0.0003028576356452486, 'reg_lambda': 0.0003997568780796951}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[29]\tvalid_0's binary_logloss: 0.149284\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[28]\tvalid_0's binary_logloss: 0.152396\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[25]\tvalid_0's binary_logloss: 0.15406\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[29]\tvalid_0's binary_logloss: 0.146821\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[27]\tvalid_0's binary_logloss: 0.151485\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:51:07,682] Trial 47 finished with value: 0.32350108322574445 and parameters: {'learning_rate': 0.007025165225350785, 'num_leaves': 138, 'max_depth': 20, 'min_child_samples': 47, 'subsample': 0.6681939219693056, 'colsample_bytree': 0.520976113611361, 'reg_alpha': 0.0008899926982751361, 'reg_lambda': 0.0001001988960292251}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[20]\tvalid_0's binary_logloss: 0.156184\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[21]\tvalid_0's binary_logloss: 0.157352\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.158353\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[23]\tvalid_0's binary_logloss: 0.153815\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[21]\tvalid_0's binary_logloss: 0.156628\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:51:09,372] Trial 48 finished with value: 0.32342878963396104 and parameters: {'learning_rate': 0.005582912854463182, 'num_leaves': 73, 'max_depth': 4, 'min_child_samples': 45, 'subsample': 0.6366754591838325, 'colsample_bytree': 0.5496525865369217, 'reg_alpha': 0.014380265524138174, 'reg_lambda': 4.523052902508379e-05}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.153275\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[19]\tvalid_0's binary_logloss: 0.155095\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[18]\tvalid_0's binary_logloss: 0.156328\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[24]\tvalid_0's binary_logloss: 0.150776\n",
      "Training until validation scores don't improve for 50 rounds\n",
      "Early stopping, best iteration is:\n",
      "[17]\tvalid_0's binary_logloss: 0.15482\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-05-12 20:51:11,194] Trial 49 finished with value: 0.32254500980011075 and parameters: {'learning_rate': 0.00782375449007001, 'num_leaves': 99, 'max_depth': 7, 'min_child_samples': 43, 'subsample': 0.5759846286909801, 'colsample_bytree': 0.6111561150708429, 'reg_alpha': 0.011360395411173944, 'reg_lambda': 0.0008835111248810218}. Best is trial 38 with value: 0.3300425481836601.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best hyperparameters:  {'learning_rate': 0.006927918069998695, 'num_leaves': 83, 'max_depth': 4, 'min_child_samples': 47, 'subsample': 0.6160943288395566, 'colsample_bytree': 0.5179983582294293, 'reg_alpha': 0.0014196850444548798, 'reg_lambda': 0.00027946017747999287}\n"
     ]
    }
   ],
   "source": [
    "def objective(trial):\n",
    "    # ハイパーパラメータをOptunaでサンプリング\n",
    "    params = {\n",
    "        'objective': 'binary',\n",
    "        'is_unbalance': True,\n",
    "        'random_state': 42,\n",
    "        'n_estimators': 1000,\n",
    "        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05,log=True),\n",
    "        'num_leaves': trial.suggest_int('num_leaves', 20, 300),\n",
    "        'max_depth': trial.suggest_int('max_depth', 3, 20),\n",
    "        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),\n",
    "        'subsample': trial.suggest_float('subsample', 0.5, 1.0),\n",
    "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),\n",
    "        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e-1,log=True),\n",
    "        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e-1,log=True)\n",
    "    }\n",
    "    \n",
    "    # StratifiedKFold 5分割\n",
    "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "    f1_scores = []\n",
    "    thresholds_list = []\n",
    "    \n",
    "    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):\n",
    "        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]\n",
    "        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]\n",
    "        \n",
    "        # モデルの学習\n",
    "        model = lgb.LGBMClassifier(**params,n_jobs=-1,verbose=-1)\n",
    "        model.fit(\n",
    "            X_train, y_train,\n",
    "            eval_set=[(X_val, y_val)],\n",
    "            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]\n",
    "        )\n",
    "        \n",
    "        y_prob = model.predict_proba(X_val)[:, 1]\n",
    "        \n",
    "        # 閾値ごとのF1スコア\n",
    "        thresholds = np.arange(0.01, 1.00, 0.01)\n",
    "        f1s = [f1_score(y_val, (y_prob > t).astype(int), zero_division=0) for t in thresholds]\n",
    "        best_threshold = thresholds[np.argmax(f1s)]\n",
    "        best_f1 = max(f1s)\n",
    "        \n",
    "        f1_scores.append(best_f1)\n",
    "        thresholds_list.append(best_threshold)\n",
    "    \n",
    "    # 最終的な平均F1スコアを返す\n",
    "    return np.mean(f1_scores)\n",
    "\n",
    "# Optunaスタディを作成\n",
    "study = optuna.create_study(direction=\"maximize\")  # F1スコアを最大化する\n",
    "study.optimize(objective, n_trials=50)  # 試行回数を50回に設定\n",
    "\n",
    "print(\"Best hyperparameters: \", study.best_params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58957d1a-a0f1-48fa-bed0-e720e23ffab0",
   "metadata": {},
   "outputs": [],
   "source": [
    "over:0.3286\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
