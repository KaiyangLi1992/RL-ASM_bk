
import sys
sys.path.extend([
        "/home/kli16/esm_NSUBS_RWSE_LapPE/esm",
        "/home/kli16/esm_NSUBS_RWSE_LapPE/esm/uclasm/",
        "/home/kli16/esm_NSUBS_RWSE_LapPE/esm/GraphGPS/",
        "/home/kli16/esm_NSUBS_RWSE_LapPE/esm/NSUBS/"
    ])

from add_noise_dataset import add_noise
from check_isomorphic import check_isomorphic
from deduplication_dataset import  deduplication_dataset
from create_feature import create_feature_LapPE,create_feature_RWSE
from create_data_imtationlearning_batch import create_batch
from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair
dataset_name = 'MSRC_21'
noiseratio = 0

# add_noise(dataset_name,noiseratio)
# check_isomorphic(dataset_name,noiseratio)
# deduplication_dataset(dataset_name,noiseratio)
create_feature_LapPE(dataset_name,'trainset',noiseratio)
create_feature_LapPE(dataset_name,'testset',noiseratio)
# create_feature_RWSE(dataset_name,'trainset',noiseratio)
# create_feature_RWSE(dataset_name,'testset',noiseratio)
create_batch(dataset_name,noiseratio)

dataset_name = 'MSRC_21'
noiseratio = 0.05

# add_noise(dataset_name,noiseratio)
# check_isomorphic(dataset_name,noiseratio)
# deduplication_dataset(dataset_name,noiseratio)
create_feature_LapPE(dataset_name,'trainset',noiseratio)
create_feature_LapPE(dataset_name,'testset',noiseratio)
# create_feature_RWSE(dataset_name,'trainset',noiseratio)
# create_feature_RWSE(dataset_name,'testset',noiseratio)
create_batch(dataset_name,noiseratio)


dataset_name = 'MSRC_21'
noiseratio = 0.1

# add_noise(dataset_name,noiseratio)
# check_isomorphic(dataset_name,noiseratio)
# deduplication_dataset(dataset_name,noiseratio)
create_feature_LapPE(dataset_name,'trainset',noiseratio)
create_feature_LapPE(dataset_name,'testset',noiseratio)
# create_feature_RWSE(dataset_name,'trainset',noiseratio)
# create_feature_RWSE(dataset_name,'testset',noiseratio)
create_batch(dataset_name,noiseratio)

