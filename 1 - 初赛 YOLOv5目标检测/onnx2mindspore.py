"""
Usage:
    $  python .\onnx2mindspore.py --weights ./runs/convert/yolov5s_datav4_1024.onnx
"""
import os
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
import argparse
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='cpp'
os.environ['HOME'] = r'C:\Users\liujin'

def parse_opt():
    parser = argparse.ArgumentParser('MindConverter')
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='path to weight, relative to proj path')
    parser.add_argument('--img_size', type=int, default=640, help='image size (default: 640)')
    parser.add_argument('--output', type=str, default='', help='output path (default is the original weigth path)')
    opt = parser.parse_args()
    opt.weights = BASE_DIR / opt.weights
    assert opt.weights.exists(), 'weight file does not exist'
    # image size
    try:
        img_size = int(opt.weights.stem.split('_')[-1])
        opt.img_size = img_size
        print(f'Change image size to {img_size}')
    except:
        print(f'Image size: {opt.img_size}')

    # output
    if opt.output=='':
        opt.output = opt.weights.parent
    else:
        opt.output = BASE_DIR / opt.output
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    cmd = f'mindconverter --model_file {str(opt.weights)} \
                --shape 1,3,{opt.img_size},{opt.img_size}  \
                --input_nodes images  \
                --output_nodes output  \
                --output {str(opt.output)}\
                --report {str(opt.output)} '
    os.system(cmd)