import os
import sys
import argparse
from PIL import Image
from PIL import ImageFilter
# import pyocr
import easyocr
from easyocr.utils import reformat_input
import glob
from tqdm import tqdm
import cv2
import numpy as np
import re

from Bio import pairwise2
from Bio.pairwise2 import format_alignment

tessdata_dir_config = '--tessdata-dir "/home/taka/work/goseki/imagerecog/tessdata_best"'

def correct_miss_num(str, list_path='../data/point_list.txt'):
    max_score = -1e3
    max_idx = 0
    out = str
    # print(str)
    
    with open(list_path) as f:
        skill_list = f.read().strip().splitlines()
        if str != '' and str not in skill_list:
            # 後ろに変なのがついてる場合('+10|')→その記号を削除
            # 数字の後ろに記号はつくはずはない→正規表現で数字以降の文字列を削除
            out = re.findall(r'[\+\-\d]+', out)[0]

            if int(out) > 13:
                # ＋→4とミスするのが多い。
                if out[0] == '4':
                # print(out, out[0])
                    out = '+' + out[1:]
                else:
                    # 後半読み込みミスしてる可能性大。後半をカット。
                    # print(out , out[:-1])
                    out = out[:-1]

            # 末尾を1とエラー読み込むパターン

            # 数字だけのとき→マイナスを読み取れてない可能性が高い
            if re.search(r'[\+\-]', out) == None:
                out = '-' + out
            
            # print(f"{str} --> {out}")
    return out

def correct_miss(str, list_path='../data/skill_list.txt'):
    max_score = -1e3
    max_idx = 0

    
    with open(list_path) as f:
        skill_list = f.read().strip().splitlines()
    
        # 特殊パターン1：アルファベットを含む文字列はなんかエラー起こすので別で検索。KO,対防御DOWNのみではある。
        if re.search(r'[a-zA-Z]', str) != None:
            # print(f'alphabet arimasu.'+str)
            if str in ['KO', '対防御DOWN']:
                return str

            # 特殊パターン2：アルファベット以外の単語も含む文字列はなんかエラー起こすので別で検索。痛a→痛撃 等。
            elif re.search(r'[^a-zA-Z]', str) != None:
                for idx, skill in enumerate(skill_list):
                    score = pairwise2.align.globalms(str, skill, 2, -1, -0.5, -0.1)[0].score
                    # print(score)
                    if score > max_score:
                        max_score = score
                        max_idx = idx
                out = skill_list[max_idx]
                return out
            else:
                # 特殊パターン2：アルファベットしか含まない文字列,かつKO以外のやつはなんかエラー起こすので削除。采配→t 等。
                # print('warning: an error has occured.' + str)
                return None
    
    # 普通パターン：日本語のスキル。
    if str != '' and str not in skill_list:
        for idx, skill in enumerate(skill_list):
            score = pairwise2.align.globalms(str, skill, 2, -1, -0.5, -0.1)[0].score
            # print(score)
            if score > max_score:
                max_score = score
                max_idx = idx
        out = skill_list[max_idx]
    else:
        out = str
            
    # print(f"{str} --> {out}")
    return out

def crop_image( path, engine, engine_num, outdir=None ):
    
    im = Image.open(path).convert('L')
    img_width, img_height = im.size
    # 護石の箇所だけクロップ
    im_crop = im.crop( ( int(img_width*0.70), int(img_height*0.14), int(img_width*0.98), int(img_height*0.54) ) )

    # # # 2値化
    # thresh = 185
    # maxval = 255
    # im_crop = np.array(im_crop)
    # im_crop = (im_crop <= thresh) * maxval
    # # print(im_crop)

    # im_crop = Image.fromarray(np.uint8(im_crop))
    # im_crop.filter(ImageFilter.MaxFilter())

    # print(f"{outdir}/crop_{os.path.basename(path)}")
    os.makedirs(f"{outdir}/crop", exist_ok=True)
    im_crop.save(f"{outdir}/crop/crop_{os.path.basename(path)}", quality=95)

    # リサイズ
    im_crop = im_crop.resize((im_crop.width * 2, im_crop.height * 2))

    # クロップしたものを9分割して護石の認識をする
    img_width, img_height = im_crop.size

    tesseract_layout=10

    # 護石の箇所だけクロップ
    for i in range(9):
        goseki = im_crop.crop( ( int(0), int(img_height*i//9), int(img_width), int(img_height*(i+1)//9) ) )
        
        goseki_width, goseki_height = goseki.size
        # print(goseki.size)
        # 各スキルの分割
        skill1 = goseki.crop( ( 0, 0, int(goseki_width*0.31), int(goseki_height) ) )
        skill1.save(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", quality=95)
        skill1_result = engine.readtext(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", output_format='dict')
        # skill1_txt = skill1_result[0]['text']
        # print(skill1_result)
        if len(skill1_result) > 0:
            skill1_txt = skill1_result[0]['text']
        else:
            skill1_txt = ''

        skill1_txt = correct_miss(skill1_txt)

        if skill1_txt == None:
            # print('an error has occured. skip.')
            continue
        if skill1_txt == '':
            # print('there is no skills. skip.')
            continue
        # print(skill1_txt)

        skill1_point = goseki.crop( ( int(goseki_width*0.31), 0, int(goseki_width*0.41), int(goseki_height) ) )
        # print(f"{outdir}/tmp_crop_{i}_{os.path.basename(path)}")
        skill1_point.save(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", quality=95)
        skill1_point_result = engine_num.readtext(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", output_format='dict')
        skill1_point_txt = skill1_point_result[0]['text']
        if len(skill1_point_result) > 0:
            skill1_point_txt = skill1_point_result[0]['text']
        else:
            skill1_point_txt = ''
        skill1_point_txt = correct_miss_num(skill1_point_txt)
        
        skill2 = goseki.crop( ( int(goseki_width*0.42), 0, int(goseki_width*0.73), int(goseki_height) ) )
        # print(f"{outdir}/tmp_crop_{i}_{os.path.basename(path)}")
        skill2.save(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", quality=95)
        skill2_result = engine.readtext(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", output_format='dict')
        if len(skill2_result) > 0:
            skill2_txt = skill2_result[0]['text']
        else:
            skill2_txt = ''
        # print(skill2_result)
        skill2_txt = correct_miss(skill2_txt)

        skill2_point = goseki.crop( ( int(goseki_width*0.73), 0, int(goseki_width*0.84), int(goseki_height) ) )
        # print(f"{outdir}/tmp_crop_{i}_{os.path.basename(path)}")
        skill2_point.save(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", quality=95)
        skill2_point_result = engine_num.readtext(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", output_format='dict')
        # print(skill2_point_result)
        if len(skill2_point_result) > 0:
            skill2_point_txt = skill2_point_result[0]['text']
        else:
            skill2_point_txt = ''
        skill2_point_txt = correct_miss_num(skill2_point_txt)
        # print(skill2_point_txt)

        # スロット→パターンマッチング http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
        slot = goseki.crop( ( int(goseki_width*0.84), 0, int(goseki_width), int(goseki_height) ) )

        # # # 2値化
        # thresh = 185
        # maxval = 255
        # slot = np.array(slot)
        # slot = (slot <= thresh) * maxval
        # # print(im_crop)

        # slot = Image.fromarray(np.uint8(slot))
        # slot.filter(ImageFilter.MaxFilter())

        slot.save(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}", quality=95)

        img_rgb = cv2.imread(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}")
        height, width, channels = img_rgb.shape[:3]
        # print( type(height//2), type(width//2), channels)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # template = cv2.imread('../data/template/template.png',0)
        template = cv2.imread('../data/template/template_jpg.png',0)
        # tmp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.97
        loc = np.where( res >= threshold)
        n_slot = len(loc[0])

        # テンプレマッチ確認用の画像出力
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        # os.makedirs(f'{outdir}/res', exist_ok=True)
        # cv2.putText(img_rgb, text=str(n_slot), org=(width//2,height//2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), lineType=cv2.LINE_4)
        # # cv2.putText(img_rgb, text=str(n_slot), org=(0,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), lineType=cv2.LINE_4)
        # cv2.imwrite(f'{outdir}/res/res_tmp_crop_{i}_{os.path.basename(path)}.png',img_rgb)

        if os.path.exists(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}"):
            os.remove(f"{outdir}/crop/tmp_crop_{i}_{os.path.basename(path)}")

        with open(f"{outdir}/goseki.csv", 'a', encoding='shift-jis') as fo:
            print(f",{n_slot},{skill1_txt},{skill1_point_txt},{skill2_txt},{skill2_point_txt}", file=fo)

        print(f",{n_slot},{skill1_txt},{skill1_point_txt},{skill2_txt},{skill2_point_txt}")

    return


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default="../out/test")
    parser.add_argument('--pngdir', type=str, default="../out/test")
    args = parser.parse_args()

    return args


def main():
    args = parser()

    os.makedirs(f"{args.pngdir}/crop", exist_ok=True)
    pnglist = glob.glob(f"{args.pngdir}/*.png")
    pnglist = glob.glob(f"{args.pngdir}/*.jpg")
    
        
    # # OCRエンジンを取得
    # engines = pyocr.get_available_tools()
    # engine = engines[0]

    engine = easyocr.Reader(['ja', 'en'], gpu=False)
    engine_num = easyocr.Reader(['en'], gpu=False, user_network_directory='/home/taka/.EasyOCR/user_network' ,recog_network='custom_example')
    # print(pnglist)

    os.makedirs(f"{args.outdir}", exist_ok=True)
    if os.path.exists(f"{args.outdir}/goseki.csv") == True :
        os.remove(f"{args.outdir}/goseki.csv")

    for pngpath in pnglist:
        crop_image(pngpath, engine, engine_num, f"{args.outdir}")

    return

if __name__ == "__main__":
    main()