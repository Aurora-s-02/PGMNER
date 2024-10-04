import json
from .processor_multiarg import MultiargProcessor


_DATASET_DIR = {
    'ace_eeqa':{
        "train_file": './data/ace_eeqa/data_final/train_convert.json',
        "dev_file": './data/ace_eeqa/data_final/dev_convert.json', 
        "test_file": './data/ace_eeqa/data_final/test_convert.json',
        "max_span_num_file": "./data/dset_meta/role_num_ace.json",
    },
    'rams':{
        "train_file": './data/RAMS_1.0/data_final/train.jsonlines',
        "dev_file": './data/RAMS_1.0/data_final/dev.jsonlines',
        "test_file": './data/RAMS_1.0/data_final/test.jsonlines',
        "max_span_num_file": "./data/dset_meta/role_num_rams.json",
    },
    "wikievent":{
        "train_file": './data/WikiEvent/data_final/train.jsonl',
        "dev_file": './data/WikiEvent/data_final/dev.jsonl',
        "test_file": './data/WikiEvent/data_final/test.jsonl',
        "max_span_num_file": "./data/dset_meta/role_num_wikievent.json",
    },
    "MLEE":{
        "train_file": './data/MLEE/data_final/train.json',
        "dev_file": './data/MLEE/data_final/train.json',
        "test_file": './data/MLEE/data_final/test.json',
        "role_name_mapping": './data/MLEE/MLEE_role_name_mapping.json',
    },
    "flickr30k_furn":{
        "train_file": './data/flickr30k_furn/data_final/train.jsonlines',
        "dev_file": './data/flickr30k_furn/data_final/dev.jsonlines',
        "test_file": './data/flickr30k_furn/data_final/test.jsonlines',
        'train_aux': 'data/flickr30k_furn/data_final/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/flickr30k_furn/data_final/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/flickr30k_furn/data_final/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/flickr30k_furn/data_final/train.pth',
        'dev_imgspath': 'data/flickr30k_furn/data_final/dev.pth',
        'test_imgspath': 'data/flickr30k_furn/data_final/test.pth',
        'image':'data/flickr30k_furn/data_final/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_furn.json'
    },
    "flickr30k_Animals":{
        "train_file": './data/Animals/train.jsonlines',
        "dev_file": './data/Animals/dev.jsonlines',
        "test_file": './data/Animals/test.jsonlines',
        'train_aux': 'data/Animals/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/Animals/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/Animals/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/Animals/train.pth',
        'dev_imgspath': 'data/Animals/dev.pth',
        'test_imgspath': 'data/Animals/test.pth',
        'image':'data/Animals/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_Animals.json',
    },
    "flickr30k_Archlnfra":{
        "train_file": './data/ArchInfra/train.jsonlines',
        "dev_file": './data/ArchInfra/dev.jsonlines',
        "test_file": './data/ArchInfra/test.jsonlines',
        'train_aux': 'data/ArchInfra/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/ArchInfra/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/ArchInfra/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/ArchInfra/train.pth',
        'dev_imgspath': 'data/ArchInfra/dev.pth',
        'test_imgspath': 'data/ArchInfra/test.pth',
        'image':'data/ArchInfra/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_ArchInfra.json',
    },
    "flickr30k_ArtDecor":{
        "train_file": './data/ArtDecor/train.jsonlines',
        "dev_file": './data/ArtDecor/dev.jsonlines',
        "test_file": './data/ArtDecor/test.jsonlines',
        'train_aux': 'data/ArtDecor/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/ArtDecor/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/ArtDecor/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/ArtDecor/train.pth',
        'dev_imgspath': 'data/ArtDecor/dev.pth',
        'test_imgspath': 'data/ArtDecor/test.pth',
        'image':'data/ArtDecor/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_ArtDecor.json',
    },
    "flickr30k_ClothingAcc":{
        "train_file": './data/ClothingAcc/train.jsonlines',
        "dev_file": './data/ClothingAcc/dev.jsonlines',
        "test_file": './data/ClothingAcc/test.jsonlines',
        'train_aux': 'data/ClothingAcc/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/ClothingAcc/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/ClothingAcc/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/ClothingAcc/train.pth',
        'dev_imgspath': 'data/ClothingAcc/dev.pth',
        'test_imgspath': 'data/ClothingAcc/test.pth',
        'image':'data/ClothingAcc/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_ClothingAcc.json',
    },
    "flickr30k_ElecEntertain":{
        "train_file": './data/ElecEntertain/train.jsonlines',
        "dev_file": './data/ElecEntertain/dev.jsonlines',
        "test_file": './data/ElecEntertain/test.jsonlines',
        'train_aux': 'data/ElecEntertain/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/ElecEntertain/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/ElecEntertain/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/ElecEntertain/train.pth',
        'dev_imgspath': 'data/ElecEntertain/dev.pth',
        'test_imgspath': 'data/ElecEntertain/test.pth',
        'image':'data/ElecEntertain/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_ElecEntertain.json',
    },
    "flickr30k_Food":{
        "train_file": './data/Food/train.jsonlines',
        "dev_file": './data/Food/dev.jsonlines',
        "test_file": './data/Food/test.jsonlines',
        'train_aux': 'data/Food/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/Food/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/Food/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/Food/train.pth',
        'dev_imgspath': 'data/Food/dev.pth',
        'test_imgspath': 'data/Food/test.pth',
        'image':'data/Food/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_Food.json'
    },
    "flickr30k_Misc":{
        "train_file": './data/Misc/dev.jsonlines',
        "dev_file": './data/Misc/dev.jsonlines',
        "test_file": './data/Misc/dev.jsonlines',
        'train_aux': 'data/Misc/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/Misc/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/Misc/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/Misc/train.pth',
        'dev_imgspath': 'data/Misc/dev.pth',
        'test_imgspath': 'data/Misc/test.pth',
        'image':'data/Misc/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_Misc.json',
    },
    "flickr30k_Nature":{
        "train_file": './data/Nature/train.jsonlines',
        "dev_file": './data/Nature/dev.jsonlines',
        "test_file": './data/Nature/test.jsonlines',
        'train_aux': 'data/Nature/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/Nature/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/Nature/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/Nature/train.pth',
        'dev_imgspath': 'data/Nature/dev.pth',
        'test_imgspath': 'data/Nature/test.pth',
        'image':'data/Nature/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_Nature.json',
    },
    "flickr30k_People":{
        "train_file": './data/People/train.jsonlines',
        "dev_file": './data/People/dev.jsonlines',
        "test_file": './data/People/test.jsonlines',
        'train_aux': 'data/People/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/People/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/People/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/People/train.pth',
        'dev_imgspath': 'data/People/dev.pth',
        'test_imgspath': 'data/People/test.pth',
        'image':'data/People/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_People.json',
    },
    "flickr30k_SignComm":{
        "train_file": './data/SignComm/train.jsonlines',
        "dev_file": './data/SignComm/dev.jsonlines',
        "test_file": './data/SignComm/test.jsonlines',
        'train_aux': 'data/SignComm/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/SignComm/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/SignComm/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/SignComm/train.pth',
        'dev_imgspath': 'data/SignComm/dev.pth',
        'test_imgspath': 'data/SignComm/test.pth',
        'image':'data/SignComm/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_SignComm.json',
    },
    "flickr30k_Sport":{
        "train_file": './data/Sport/train.jsonlines',
        "dev_file": './data/Sport/dev.jsonlines',
        "test_file": './data/Sport/test.jsonlines',
        'train_aux': 'data/Sport/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/Sport/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/Sport/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/Sport/train.pth',
        'dev_imgspath': 'data/Sport/dev.pth',
        'test_imgspath': 'data/Sport/test.pth',
        'image':'data/Sport/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_Sport.json',
    },
    "flickr30k_TransVeh":{
        "train_file": './data/TransVeh/train.jsonlines',
        "dev_file": './data/TransVeh/dev.jsonlines',
        "test_file": './data/TransVeh/test.jsonlines',
        'train_aux': 'data/TransVeh/flickr30k-images_aux_images/train/crops',
        'dev_aux': 'data/TransVeh/flickr30k-images_aux_images/dev/crops',
        'test_aux': 'data/TransVeh/flickr30k-images_aux_images/test/crops',
        'train_imgspath': 'data/TransVeh/train.pth',
        'dev_imgspath': 'data/TransVeh/dev.pth',
        'test_imgspath': 'data/TransVeh/test.pth',
        'image':'data/TransVeh/flickr30k-images',
        "role_name_mapping": './data/dset_meta/role_num_flickr30k_TransVeh.json',
    },
    "flickr30k_test_food":{
        "train_file": './data/test_food/train.jsonlines',
        "dev_file": './data/test_food/dev.jsonlines',
        "test_file": './data/test_food/test.jsonlines',
        "role_name_mapping": './data/dset_meta/role_num_Food.json',

    },
}


def build_processor(args, tokenizer, transform):
    if args.dataset_type not in _DATASET_DIR: 
        raise NotImplementedError("Please use valid dataset name")
    args.train_file=_DATASET_DIR[args.dataset_type]['train_file']
    args.dev_file = _DATASET_DIR[args.dataset_type]['dev_file']
    args.test_file = _DATASET_DIR[args.dataset_type]['test_file']
    args.train_aux = _DATASET_DIR[args.dataset_type]['train_aux']
    args.dev_aux = _DATASET_DIR[args.dataset_type]['dev_aux']
    args.test_aux = _DATASET_DIR[args.dataset_type]['test_aux']
    args.train_imgspath = _DATASET_DIR[args.dataset_type]['train_imgspath']
    args.dev_imgspath = _DATASET_DIR[args.dataset_type]['dev_imgspath']
    args.test_imgspath = _DATASET_DIR[args.dataset_type]['test_imgspath']
    args.image = _DATASET_DIR[args.dataset_type]['image']

    args.role_name_mapping = None
    if args.dataset_type=="MLEE":
        with open(_DATASET_DIR[args.dataset_type]['role_name_mapping']) as f:
            args.role_name_mapping = json.load(f)

    processor = MultiargProcessor(args, tokenizer, transform)
    return processor

