import pandas as pd
from sklearn.model_selection import train_test_split


path = '../data/cloud_data'

train = pd.read_csv(f'{path}/train.csv')
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])


id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label']    .apply(lambda x: x.split(
    '_')[0])\
    .value_counts().reset_index().rename(columns={'index': 'img_id',
                                                  'Image_Label': 'count'})
for i in range(5):
    train_ids, valid_ids = train_test_split(
        id_mask_count['img_id'].values,
        random_state=i,
        stratify=id_mask_count['count'], test_size=0.1)
    df_t = pd.DataFrame()
    df_v = pd.DataFrame()
    df_t['train_ids'] = train_ids
    df_v['valid_ids'] = valid_ids

    df_t.to_csv(f'./folds/fold_{i+1}_train.csv', index=False)
    df_v.to_csv(f'./folds/fold_{i+1}_val.csv', index=False)
