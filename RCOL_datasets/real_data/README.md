# Please download the datasets from the following links:

Power - https://archive.ics.uci.edu/ml/datasets/Power+consumption+of+Tetouan+city 

Energy - https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

Traffic - https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume

Wind - https://www.kaggle.com/datasets/l3llff/wind-power

Prices - https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries/blob/main/data_prices/Prices_2016_2019_extract.csv


# To download KITTI dataset follow these steps:

1. Enter this [link](http://www.cvlibs.net/datasets/kitti/raw_data.php).
2. Download the rgb "synced+rectified" data from date 2011/09/26 of indexes:
   1. 0051
   2. 0057
   3. 0059
   4. 0096
   5. 0104
3. Download the rgb "synced+rectified" data from date 2011/09/29 of index 0071
4. Download the sparse depth maps from [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip).
5. Organize the data according to the structure below. 
6. Run depths/KITTI/make_full_kitti_depths.py from to generate a full depth map from the sparse one (this may take several hours)
7. Run depths/KITTI/make_data_json.py

```
|--KITTI
|   |--rgbs
|   |   |--2011_09_26
|   |   |   |--2011_09_26_drive_0051_sync
|   |   |   |--2011_09_26_drive_0057_sync
|   |   |   |--2011_09_26_drive_0059_sync
|   |   |   |--2011_09_26_drive_0096_sync
|   |   |   |--2011_09_26_drive_0104_sync
|   |   |--2011_09_29
|   |   |   |--2011_09_29_drive_0071_sync
|   |--depths
|   |   |--2011_09_26
|   |   |   |--2011_09_26_drive_0051_sync
|   |   |   |--2011_09_26_drive_0057_sync
|   |   |   |--2011_09_26_drive_0059_sync
|   |   |   |--2011_09_26_drive_0096_sync
|   |   |   |--2011_09_26_drive_0104_sync
|   |   |--2011_09_29
|   |   |   |--2011_09_29_drive_0071_sync
```
