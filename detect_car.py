# -*- coding: utf-8 -*-
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import tensorflow
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from homography import *
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
# from tensorflow.python.compiler.mlcompute import mlcompute

# # Select CPU device.s
# mlcompute.set_mlc_device(device_name='cpu')  
# Available options are 'cpu', 'gpu', and 'any'

def check_inout(img, data_list, point_list):
    center_point_x = img.shape[0]/2

    inout_list = []

    for i in point_list:

        for j in data_list:
            if j[0]<center_point_x:
                xy = [j[2],j[3]]
            else:
                xy = [j[0],j[3]]
            if (xy[0] > i[0]) and (xy[0] < i[2]) and (xy[1] < i[3]) and (xy[1] > i[1]):
                a = 1
                break
            else:
                a = 0
        inout_list.append(a)
        
    print(inout_list)
    return inout_list

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    #
    # upload vedio
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 디렉토리 정의
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 초기화
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 모델 로드하기
    model = attempt_load(weights, map_location=device)  # load FP32 model
    #몇칸씩 필터가 움직일건지
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    ###################################################################################################################################3
    if half:
        model.half()  # to FP16

    # Second-stage classifier/분류기?
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader/데이터 로더
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    data_list = []

    count = 0  # threshold 값을 바꿔주기 위한 변수, 2일 경우 absdiff threshold 값으로 바꿔줌
    abs_conf = opt.abs_conf  # absdiff yolo 정확도 상수


#######################################################################################################################################
    for path, img, im0s, vid_cap in dataset:
        
        # print("vid cap is ", vid_cap) none으로
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 이미지 정규화
        if img.ndimension() == 3: #RGB채널? : 몇차원인지, 3차원이면
            img = img.unsqueeze(0) #첫번째 차원에 1인 차원이 생성
            # print(img)
        # print(img.shape)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS

        if count == 2:
            pred = non_max_suppression(pred, abs_conf, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        else:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

        #NMS 를 사용해서 연산량, iou 최적화

        t2 = time_synchronized()
        #t1,t2 시간 변수를 생성하여 시간 차이 프린트
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


#####################################################################################################################################################################

        # Process detections
        for i, det in enumerate(pred):  # detections per image 
            
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count #변수할당
                # print("if : im0s : ",im0s)
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0) #변수할당
                # print("else : im0s : ",im0s)

            p = Path(p)  # to Path
            print("p is ", p)
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #im0 정규화  ㅣ: gn

            dat = [] 

            if len(det):
                # Rescale boxes from img_size to im0 size
                #
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() #두번째 행, 모든행과 4번째 열까지
                #왼쪽 위 모서리, 오른쪽 아래 모서리 좌표
                #[:,-1] 의미 : 모든 행과 마지막 열
                # Print results
                for c in det[:, -1].unique(): 
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #print(*xyxy)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        print("if에 들어가서", xywh)
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image #이미지에 바운딩 박스 그리기
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        if names[int(cls)] == 'car': #클래스가 CAR일 경우
                            label = f'{names[int(cls)]} {conf:.2f}' #confidence
                            xyxy_ = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])] #픽셀 좌표
                            dat.append(xyxy_) #dat리스트에 저장
                            # x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, (int(xyxy[1]) + int(xyxy[3])) / 2  # 중심점 좌표 x, y
                            #  크기? 중심점 위치? 를 사용하여 디텍팅 되는 차량을 한정지음

                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            data_list.append(dat)# dat리스트를 추가

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)') #시간 계산
            
            after_tmp = opt.after
            frame_tmp = cv2.imread(after_tmp)  # 현재 사진 로딩
            frame_tmp = cv2.resize(frame_tmp, dsize=(720,480))
            data_list_tmp = np.array(dat)
            data_list_tmp = data_list_tmp.squeeze()

            count_q =0
            for x1,y1,x2,y2 in data_list_tmp:
                resize_img = frame_tmp[y1:y2, x1:x2, :]
                resize_img = cv2.resize(resize_img, dsize=(720,480))
                cv2.imshow('image', resize_img)
                k = cv2.waitKey(0)
                if k == 27:
                    cv2.destroyAllWindows()
                elif k == ord('s'):
                    cv2.imwrite('./save_img/saveimg{}.png'.format(count_q), resize_img)
                    cv2.destroyAllWindows()
                count_q += 1

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                print("view_img is true")

            # Save results (image with detections)
            if save_img: #이미지를 저장
                # print("save_img is ture")
                if dataset.mode == 'image': #이게 이미지일 경우
                    # print("datasetmode is image")
                    cv2.imwrite(save_path, im0) #save_path에 im0저장
                else:  # 'video' or 'stream'
                    print("vid_path is ", vid_path) #사전에 none이라고 정의 60번째 줄
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter): #int형인지 체크 
                            # print("낙규낙규낙규")
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        count = count + 1
        print(count)
###############################################################################################################################################

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")



    iou_list = []  # abs 와 iou 비교후 일치하는 차량 박스 리스트
    after_count = 0

    # # 현재 사진과 absdiff iou 리스트 비교

    for i in iou_list:
        for k in data_list[1]:
            if k == i:
                after_count = after_count + 1

    #  absdiff 이미지 차량 박스 검출 후 저장
    after = opt.after
    frame_ = cv2.imread(after)  # 현재 사진 로딩
    frame_ = cv2.resize(frame_, dsize=(720,480))

    for i in iou_list:
        label = f'car'
        plot_one_box(i, frame_, label=label, line_thickness=2)

    cv2.imwrite('data/images/def/abs.png', frame_)

    cv2.imshow('res', frame_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'Done. ({time.time() - t0:.3f}s)')
###############################################################################################################################################


def get_iou(a, b, epsilon=1e-5):  # bounding box 교차 영역 체크
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if width < 0 or height < 0:
        return 0.0
    area_overlap = width * height
    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def absdiff():
    before, after = opt.before, opt.after
    
    frame_ = cv2.imread(before)
    frame_ = cv2.resize(frame_, dsize=(720,480))
    # hsv_frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2HSV)
    # absdiff_frame = np.zeros(frame_.shape[:2], np.uint8)

    ground = cv2.imread(after)
    ground = cv2.resize(ground, (720, 480))

    mask_frame = ground
    absdiff_frame = cv2.absdiff(mask_frame, frame_)
    # cv2.imshow('res', absdiff_frame)
    cv2.imwrite('data/images/test4.png', absdiff_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images/input_image', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--before', type=str, default='data/images/def/12_5.png', help='before images')
    parser.add_argument('--after', type=str, default='data/images/def/cam80.png', help='after images')
    parser.add_argument('--abs-conf', type=float, default=0.1, help='absdiff threshold')
    parser.add_argument('--abs-iou', type=float, default=0.75, help='iou threshold')
    opt = parser.parse_args()
    print("opt is ", opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                
                detect()
                strip_optimizer(opt.weights)
        else:
            absdiff()
            detect()
