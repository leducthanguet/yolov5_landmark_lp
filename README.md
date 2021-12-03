# yolov5_landmark_lp
- This is a project about detect license plates for truck of Viet Nam.
- To use code you need:
  + git clone  https://github.com/leducthanguet/yolov5_landmark_lp.git
  + cd yolov5_landmark_lp
  + pip install -r requirements.txt
- To train custom dataset you need create dataset with format yolo_v5. But label.txt for per line is x y w h x1 y1 x2 y2 x3 y3 x4 y4
  + Use command line :
    python3 train.py --batch-size=32 --epoch=200 --saving_epoch=125 --data=lp.yaml --hyp=hyp.lp.yaml --weights=weights/yolov5s.pt --cfg=models/yolov5s2.yaml --cache-images
