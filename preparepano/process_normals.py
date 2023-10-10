import argparse
import os
import sys
import numpy as np
from PIL import Image
import tqdm
import cv2

def process_normals(base_dir: str, scan_id: str):
    srcdir = os.path.join(base_dir, scan_id, scan_id, 'undistorted_normal_images')

    filelist = sorted(os.listdir(srcdir))
    filedict = {}
    for filename in filelist:
        namepart, _ = os.path.splitext(filename)
        tokens = namepart.split("_")
        assert len(tokens) == 4

        locationId = tokens[0]
        horizonId = tokens[1]
        cameraId = tokens[2]
        normalId = tokens[3]
        
        if normalId in ['nx', 'ny', 'nz']:
            srcimg = np.array(Image.open(os.path.join(srcdir, filename)))
            srcimg = (srcimg / 32768.0) * 127.5 # 16 bit -> 8 bit
        
            if not(locationId) in filedict.keys():
                filedict[locationId] = {}
                filedict[locationId][horizonId] = {}
                filedict[locationId][horizonId][cameraId] = {}
                filedict[locationId][horizonId][cameraId][normalId] = srcimg
            elif not(horizonId) in filedict[locationId].keys():
                filedict[locationId][horizonId] = {}
                filedict[locationId][horizonId][cameraId] = {}
                filedict[locationId][horizonId][cameraId][normalId] = srcimg
            elif not(cameraId) in filedict[locationId][horizonId].keys():
                filedict[locationId][horizonId][cameraId] = {}
                filedict[locationId][horizonId][cameraId][normalId] = srcimg
            elif not(normalId) in filedict[locationId][horizonId][cameraId].keys():
                filedict[locationId][horizonId][cameraId][normalId] = srcimg
            else:
                raise ValueError("Error")
        else:
            continue

    destdir = os.path.join(base_dir, scan_id, scan_id, 'undistorted_normal_images_processed')
    if not(os.path.exists(destdir)):
        os.mkdir(destdir)

    for locationId in filedict.keys():
        for horizonId in filedict[locationId]:
            for cameraId in filedict[locationId][horizonId]:
                # nx ny nz -> RGB
                destimg = cv2.merge((filedict[locationId][horizonId][cameraId]['nx'],
                                     filedict[locationId][horizonId][cameraId]['ny'],
                                    filedict[locationId][horizonId][cameraId]['nz']))
                eqrimg = Image.fromarray(destimg.astype(np.uint8))
                eqrimg.save(os.path.join(destdir, locationId + "_" + horizonId + "_" + cameraId + ".png"))

def parse_arguments(args):
    usage_text = (
        "Matterport3D preprocessing script"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("--m3d_path", type=str, 
        help="Input Matterport3D root path"
    )
    parser.add_argument("--scan_id", type=str,
        help="Process single specified scan rather than getting list of all scans"
    )
    parser.add_argument("--all_train_scans", action="store_true",
        help="Process all scans of the train set rather than getting list of all scans"
    )
    parser.add_argument("--all_validation_scans", action="store_true",
        help="Process all scans of the validation set rather than getting list of all scans"
    )
    parser.add_argument("--all_test_scans", action="store_true",
        help="Process all scans of the test set rather than getting list of all scans"
    )
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, _ = parse_arguments(sys.argv)

    scan_id_list = []
    if not(args.scan_id==None):
        scan_id_list = tqdm.tqdm([args.scan_id], desc="Dataset Progress")
    elif args.all_train_scans:
        train_id_list = [
            '17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '2azQ1b91cZZ', '2n8kARJN3HM',
            '5LpN3gDmAk7', '5ZKStnWn8Zo', '5q7pvUzZiYa', '759xd9YjKW5', '8194nk5LbLH', '82sE5b5pLXE',
            '8WUmhLawc2A', 'ARNzJeq3xxb', 'D7N2EKCX4Sj', 'E9uDoFAP3SH', 'EDJbREhghzL', 'EU6Fwq7SyZv',
            'JF19kD82Mey', 'JmbYfDe2QKZ', 'PX4nDJXEHrG', 'PuKPg4mmafe', 'QUCTc6BB5sX', 'SN83YJsR3w2',
            'TbHJrupSAjP', 'ULsKaCPVFJR', 'V2XKFyX4ASd', 'VVfe2KiqLaN', 'Vt2qJdWjCF2', 'Vvot9Ly1tCj',
            'VzqfbhrpDEA', 'WYY7iVyf5p8', 'XcA2TqTSSAj', 'YFuZgdQ5vWj', 'YVUC4YcDtcY', 'aayBHfsNo7d',
            'ac26ZMwG7aT', 'cV4RVeZvu5T', 'fzynW3qQPVF', 'gTV8FGcVJC9', 'gxdoqLR6rwA', 'gZ6f7yhEvPG',
            'i5noydFURQK', 'mJXqzFtmKg4', 'oLBMNvg9in8', 'p5wJjkQkbXX', 'qoiz87JEwZ2', 'r1Q1Z4BcV1o',
            'r47D5H71a5s', 's8pcmisQ38h', 'sKLMLpTHeUy', 'sT4fr6TAbpF', 'ur6pFq6Qu1A', 'vyrNrziPKCB',
            'Uxmj2M2itWa', 'RPmz2sHmrrY', 'Pm6F8kyY3z2', 'pLe4wQe7qrG', 'JeFG25nYj2p', 'HxpKQynjfin',
            '7y3sRwLe3Va', '2t7WUuJeko7', 'B6ByNegPMKs', 'S9hNv5qa7GM', 'zsNo4HB9uLZ', 'kEZ7cmS4wCh'
        ]
    elif args.all_validation_scans:
        validation_id_list = [
            'UwV83HsGsw3', 'X7HyMhZNoso', 'Z6MFQCViBuw', 'b8cTxDM8gDG', 'e9zR4mvMWw7', 'q9vSo1VnCiC',
            'rPc6DW4iMge', 'rqfALeAoiTq', 'uNb9QFRL6hY', 'wc2JMjhGNzB', 'x8F5xyUWy9e', 'yqstnuAEVhm'
        ]
    elif args.all_test_scans:
        test_id_list = [
            'VFuaQ6m2Qom', 'VLzqgDo317F', 'ZMojNkEp431', 'jh4fc5c5qoQ', 'jtcxE69GiFV', 'pRbA3pwrgk9',
            'pa4otMbVnkk', 'D7G3Y4RVNrH', 'dhjEzFoUFzH', 'GdvgFV5R1Z5', 'gYvKGZ5eRqb', 'YmJkqBEsHnH'
        ]
        scan_id_list = tqdm.tqdm(test_id_list, desc="Dataset Progress")
    else: 
        scan_id_list = tqdm.tqdm(os.listdir(args.m3d_path), desc="Dataset Progress")

    for scan_id in scan_id_list:
        process_normals(base_dir=args.m3d_path, scan_id=scan_id)