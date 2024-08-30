### --------------- Configuring the Train and Valid datasets ---------------

# MagneticTile_train = {"name": "MagneticTile_train", # the evaluation of this dataset has some problem
#                 "im_dir": "/data/hdc/jinglong/datasets/MagneticTile_merged/train",
#                 "gt_dir": "/data/hdc/jinglong/datasets/MagneticTile_merged/train",
#                 "im_ext": ".jpg",
#                 "gt_ext": ".png"}

# ----------------------------------------- 1013 added ---------------------------------------
BSData_train = {"name": "BSData_train",
				"im_dir": "/data/hdc/jinglong/datasets/BSData-main/my_train",
				"gt_dir": "/data/hdc/jinglong/datasets/BSData-main/my_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

RSDD_I_train = {"name": "RSDD_I_train",
				"im_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-I RSDDs dataset/my_train",
				"gt_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-I RSDDs dataset/my_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

RSDD_II_train = {"name": "RSDD_II_train",
				"im_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-II RSDDs dataset/my_train",
				"gt_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-II RSDDs dataset/my_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

# ----------------------------------------- mvtec 3 groups ---------------------------------------
group1_train = {"name": "group1",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group1/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group1/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
group2_train = {"name": "group2",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group2/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group2/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
group3_train = {"name": "group3",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group3/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group3/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
# ----------------------------------------- mvtec 10 object ---------------------------------------
# [carpet, leather, tile, bottle, capsule, metal_nut, pill, screw, transistor, zipper]

carpet_train = {"name": "carpet_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/carpet/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/carpet/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
leather_train = {"name": "leather_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/leather/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/leather/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
tile_train = {"name": "tile_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/tile/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/tile/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
bottle_train = {"name": "bottle_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/bottle/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/bottle/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
capsule_train = {"name": "capsule_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/capsule/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/capsule/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
metal_nut_train = {"name": "metal_nut_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/metal_nut/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/metal_nut/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
pill_train = {"name": "pill_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/pill/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/pill/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
screw_train = {"name": "screw_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/screw/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/screw/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
transistor_train = {"name": "transistor_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/transistor/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/transistor/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
zipper_train = {"name": "zipper_train",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/zipper/new_train",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/zipper/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

# --------------------------------------------------------------------------------
MetalSurfaceDefect_train = {"name": "MetalSurfaceDefect_train",
				"im_dir": "/data/hdc/jinglong/datasets/MetalSurfaceDefect东北大学_merged/train",
				"gt_dir": "/data/hdc/jinglong/datasets/MetalSurfaceDefect东北大学_merged/train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

CAMO_train = {"name": "CAMO_train",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/CAMO迷彩动物分割/CAMO-COCO-V.1.0-CVIU2019/Camouflage/Images/Train",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/CAMO迷彩动物分割/CAMO-COCO-V.1.0-CVIU2019/Camouflage/GT",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

ISTR_train = {"name": "ISTR_train",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/ISTD_Dataset/train/train_A",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/ISTD_Dataset/train/train_B",
				"im_ext": ".png",
				"gt_ext": ".png"}

Polyp_train = {"name": "Polyp_train",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/kvasir_polyp分割/train/images",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/kvasir_polyp分割/train/masks",
				"im_ext": ".jpg",
				"gt_ext": ".jpg"}

# valid set

# MagneticTile_valid = {"name": "MagneticTile_valid", # the evaluation of this dataset has some problem
#                 "im_dir": "/data/hdc/jinglong/datasets/MagneticTile_merged/valid",
#                 "gt_dir": "/data/hdc/jinglong/datasets/MagneticTile_merged/valid",
#                 "im_ext": ".jpg",
#                 "gt_ext": ".png"}

# ----------------------------------------- 1013 added ---------------------------------------
BSData_test = {"name": "BSData_test",
				"im_dir": "/data/hdc/jinglong/datasets/BSData-main/my_test",
				"gt_dir": "/data/hdc/jinglong/datasets/BSData-main/my_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

RSDD_I_test = {"name": "RSDD_I_test",
				"im_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-I RSDDs dataset/my_test",
				"gt_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-I RSDDs dataset/my_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

RSDD_II_test = {"name": "RSDD_II_test",
				"im_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-II RSDDs dataset/my_test",
				"gt_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-II RSDDs dataset/my_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

# ----------------------------------------- mvtec 3 groups ---------------------------------------
group1_test = {"name": "group1",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group1/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group1/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
group2_test = {"name": "group2",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group2/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group2/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
group3_test = {"name": "group3",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group3/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group3/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

# ----------------------------------------- mvtec 10 object ---------------------------------------
carpet_test = {"name": "carpet_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/carpet/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/carpet/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
leather_test = {"name": "leather_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/leather/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/leather/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
tile_test = {"name": "tile_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/tile/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/tile/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
bottle_test = {"name": "bottle_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/bottle/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/bottle/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
capsule_test = {"name": "capsule_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/capsule/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/capsule/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
metal_nut_test = {"name": "metal_nut_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/metal_nut/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/metal_nut/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
pill_test = {"name": "pill_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/pill/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/pill/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
screw_test = {"name": "screw_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/screw/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/screw/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
transistor_test = {"name": "transistor_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/transistor/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/transistor/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
zipper_test = {"name": "zipper_test",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec/zipper/new_test",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec/zipper/new_test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
# --------------------------------------------------------------------------------
MetalSurfaceDefect_test = {"name": "MetalSurfaceDefect_test",
				"im_dir": "/data/hdc/jinglong/datasets/MetalSurfaceDefect东北大学_merged/test",
				"gt_dir": "/data/hdc/jinglong/datasets/MetalSurfaceDefect东北大学_merged/test",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

CAMO_test = {"name": "CAMO_test",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/CAMO迷彩动物分割/CAMO-COCO-V.1.0-CVIU2019/Camouflage/Images/Test",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/CAMO迷彩动物分割/CAMO-COCO-V.1.0-CVIU2019/Camouflage/GT",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

ISTR_test = {"name": "ISTR_test",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/ISTD_Dataset/test/test_A",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/ISTD_Dataset/test/test_B",
				"im_ext": ".png",
				"gt_ext": ".png"}

Polyp_test = {"name": "Polyp_test",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/kvasir_polyp分割/sessile-main-Kvasir-SEG/images",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/kvasir_polyp分割/sessile-main-Kvasir-SEG/masks",
				"im_ext": ".jpg",
				"gt_ext": ".jpg"}

coco_test = {"name": "coco_test",
				"im_dir": "/data/hdc/jinglong/coco/evaluateSAM_COCO",
				"gt_dir": "/data/hdc/jinglong/coco/evaluateSAM_COCO",
				"im_ext": ".jpg",
				"gt_ext": ".png"}


# ----------------------------------------- 各个数据集保存的memory bank ------------------------------------------
BSData_membank = {"name": "BSData_membank",
				"im_dir": "/data/hdc/jinglong/datasets/BSData-main/my_membank",
				"gt_dir": "/data/hdc/jinglong/datasets/BSData-main/my_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

RSDD_I_membank = {"name": "RSDD_I_membank",
				"im_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-I RSDDs dataset/my_membank",
				"gt_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-I RSDDs dataset/my_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

RSDD_II_membank = {"name": "RSDD_II_membank",
				"im_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-II RSDDs dataset/my_membank",
				"gt_dir": "/data/hdc/jinglong/datasets/RSDDs 数据集/Type-II RSDDs dataset/my_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

group1_membank = {"name": "group1",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group1/new_membank",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group1/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
group2_membank = {"name": "group2",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group2/new_membank",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group2/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}
group3_membank = {"name": "group3",
				"im_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group3/new_membank",
				"gt_dir": "/data/hdc/jinglong/datasets/mvtec_groups/group3/new_train",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

CAMO_membank = {"name": "CAMO_membank",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/CAMO迷彩动物分割/CAMO-COCO-V.1.0-CVIU2019/Camouflage/Images/membank",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/CAMO迷彩动物分割/CAMO-COCO-V.1.0-CVIU2019/Camouflage/GT",
				"im_ext": ".jpg",
				"gt_ext": ".png"}

ISTR_membank = {"name": "ISTR_membank",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/ISTD_Dataset/train/train_A_membank",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/ISTD_Dataset/train/train_B",
				"im_ext": ".png",
				"gt_ext": ".png"}

Polyp_membank = {"name": "Polyp_membank",
				"im_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/kvasir_polyp分割/train/images_membank",
				"gt_dir": "/data/hdc/jinglong/datasets/SAM_Adapter数据集/kvasir_polyp分割/train/masks",
				"im_ext": ".jpg",
				"gt_ext": ".jpg"}



# train_datasets = [MetalSurfaceDefect_train, CAMO_train, ISTR_train, Polyp_train]
# valid_datasets = [MetalSurfaceDefect_test, CAMO_test, ISTR_test, Polyp_test]

# train_datasets = [carpet_train, leather_train, tile_train, bottle_train, capsule_train, metal_nut_train, pill_train, screw_train, transistor_train, zipper_train]
# valid_datasets = [carpet_test, leather_test, tile_test, bottle_test, capsule_test, metal_nut_test, pill_test, screw_test, transistor_test, zipper_test]

# train_datasets = [group1_train, group2_train, group3_train, MetalSurfaceDefect_train, CAMO_train, ISTR_train, Polyp_train]
# valid_datasets = [group1_test, group2_test, group3_test, MetalSurfaceDefect_test, CAMO_test, ISTR_test, Polyp_test]

train_datasets = [BSData_train, RSDD_I_train, RSDD_II_train, group1_train, group2_train, group3_train, CAMO_train, Polyp_train]
valid_datasets = [coco_test, BSData_test, RSDD_I_test, RSDD_II_test, group1_test, group2_test, group3_test, CAMO_test, Polyp_test]
membank_datasets = [BSData_membank, RSDD_I_membank, RSDD_II_membank, group1_membank, group2_membank, group3_membank, CAMO_membank, Polyp_membank]

train_datasets_reverse = [Polyp_train, CAMO_train, group1_train, group2_train, group3_train, RSDD_I_train, RSDD_II_train, BSData_train]
train_datasets_pos2 = [RSDD_I_train, RSDD_II_train, BSData_train, group1_train, group2_train, group3_train, CAMO_train, Polyp_train]
train_datasets_pos3 = [RSDD_I_train, RSDD_II_train, group1_train, group2_train, group3_train, BSData_train, CAMO_train, Polyp_train]
train_datasets_pos4 = [RSDD_I_train, RSDD_II_train, group1_train, group2_train, group3_train, CAMO_train, BSData_train, Polyp_train]
train_datasets_pos5 = [RSDD_I_train, RSDD_II_train, group1_train, group2_train, group3_train, CAMO_train, Polyp_train, BSData_train]

trainsets_order_map = {'train_datasets': train_datasets,
					   'train_datasets_reverse': train_datasets_reverse,
					   'train_datasets_pos2': train_datasets_pos2,
					   'train_datasets_pos3': train_datasets_pos3,
					   'train_datasets_pos4': train_datasets_pos4,
					   'train_datasets_pos5': train_datasets_pos5}