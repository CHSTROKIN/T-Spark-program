import argparse
from hand_track.hand_track import hand_track_realtime

if __name__ == '__main__':
    freq = 5
    width = 480
    height = 640
    freq_label = 15
    interrupt_when_lost = True

    parser = argparse.ArgumentParser(description='PyTorch Gesture & Hand Pose Compound Test')
    # config for recog
    parser.add_argument("--config", type=str, default="gesture_recog/configs/R22.py")

    parser.add_argument('--model_path', type=str,
                        default='hand_track/ckpt/shufflenet_v2_x1_5-size-128-model_epoch-40.pth',
                        help='model_path')  # 模型路径
    parser.add_argument('--model', type=str, default='shufflenet_v2_x1_5',
                        help='' '''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
             shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,ReXNetV1''')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42,
                        help='num_classes')  # 手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default='0,1',
                        help='GPUS')  # GPU选择
    parser.add_argument('--test_path', type=str, default='./test_p21/frames/',
                        help='test_path')  # 测试图片路径
    parser.add_argument('--img_size', type=int, default=128,
                        help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--vis', type=bool, default=True,
                        help='vis')  # 是否可视化图片
    parser.add_argument('--out', type=str, default='./test_p21/frames_out/', help='out_file')
    parser.add_argument("--detection_thresh", type=float, default=0.7)
    parser.add_argument("--tracking_thresh", type=float, default=0.7)
    hand_track_realtime(parser, freq=freq, width=width, height=height, freq_label = freq_label, interrupt_when_lost = interrupt_when_lost)