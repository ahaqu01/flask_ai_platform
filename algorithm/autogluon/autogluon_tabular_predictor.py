from autogluon.tabular import TabularDataset,TabularPredictor, predictor
import pandas as pd
import numpy as np

# 训练
train_data = TabularDataset('train.csv')
id, label = 'Id', 'SalePrice'

# # 数据预处理
# large_val_cols = ['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','WoodDeckSF','OpenPorchSF',
#                   'EnclosedPorch','3SsnPorch','MiscVal','MoSold','YrSold']
# for c in large_val_cols + [label]:
#     train_data[c] = np.log(train_data[c]+1)

# autogluon可以做特征抽取，但适当加入一些人工预处理
# 使用multimodel这个选项来使用transformer来提取特征，并做多模型融合
# 然后做多层模型ensemble来得到更好精度
predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id])) # 无multimodel
# predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]),
#                                               hyperparams = "multimodel",
#                                               num_stack_levels=1,
#                                               num_bag_folds =5)

# 预测
test_data = TabularDataset("test.csv" )
preds = predictor.predict(test_data.drop(columns=[id]))
submission = pd.DataFrame({id:test_data[id], label:preds})
submission.to_csv("submission.csv", index=False)