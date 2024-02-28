import cv2
import numpy as np
import onnxruntime as ort
import time
import smbus
import sys
sys.path.append("..")
from cgq import mlx
import RPi.GPIO as GPIO

if __name__ == "__main__":
    
    # 模型加载
    model_pb_path = "mask.onnx"
    so = ort.SessionOptions()
    net = ort.InferenceSession(model_pb_path, so)
    
    model_pb_path1 = "nose.onnx"
    so1 = ort.SessionOptions()
    net1 = ort.InferenceSession(model_pb_path1, so1)
    
    # 标签字典
    dic_labels= {0:'with_mask',
            1:'without_mask'
            }
    
    dic_labels1= {0:'withmask_error'}
    
    # 模型参数
    model_h = 320
    model_w = 320
    nl = 3
    na = 3
    stride=[8.,16.,32.]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
    
    t3 = 0
    video = 0
    flag_msk = 0
    
    servo_SIG = 32
    pin_sig = 15
    servo_VCC = 4
    servo_GND = 6
    servo_freq = 50
    servo_time = 0.01
    servo_width_min = 2.5
    servo_width_max = 12.5
    
    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(servo_SIG, GPIO.OUT)
    servo = GPIO.PWM(servo_SIG, servo_freq)  
    servo.start(7.5)
    print('预设置完成，两秒后开始摆动')

    GPIO.setup(pin_sig,GPIO.OUT)
    GPIO.output (pin_sig,GPIO.LOW)
    sensor = mlx.MLX90614()
    
    cap = cv2.VideoCapture(video)
    img1 = np.zeros((512,642,3), np.uint8)
    
    try:
        
        while True:
            img0 = img1
            cv2.imshow("video",img0)
            key=cv2.waitKey(1)
            servo.stop()
            GPIO.output(pin_sig,GPIO.LOW)
            temp=sensor.get_obj_temp()
            print(temp)
            
            if temp>36 and temp<37.5:
                while t3<4:
                    success, img0 = cap.read()
                    if success:            
                        t1=time.time()
                        det_boxes,scores,ids = mlx.infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5)
                        if ids==0:
                            det_boxes1,scores1,ids1 = mlx.infer_img(img0,net1,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5)
                            if ids1==0 and scores1>=0.6:
                                for box,score,id in zip(det_boxes,scores,ids):
                                    break
                                for score,id in zip(scores1,ids1):
                                    break
                                label1 = '%s:%.2f'%(dic_labels1[id],score)
                                mlx.plot_one_box(box.astype(np.int16), img0, color=(255,0,0), label=label1, line_thickness=None)
                                flag_msk=2
                            else:
                                for box,score,id in zip(det_boxes,scores,ids):
                                    break
                                label = '%s:%.2f'%(dic_labels[id],score)
                                mlx.plot_one_box(box.astype(np.int16), img0, color=(255,0,0), label=label, line_thickness=None)
                                flag_msk=1
                        elif ids==1:
                            for box,score,id in zip(det_boxes,scores,ids):
                                break
                            label = '%s:%.2f'%(dic_labels[id],score)
                            mlx.plot_one_box(box.astype(np.int16), img0, color=(255,0,0), label=label, line_thickness=None)
                            flag_msk=3
                    t2=time.time()
                    
                    t3=t3+(t2-t1)
                    str_FPS = "FPS: %.2f"%(1./(t2-t1))
                    cv2.putText(img0,str_FPS,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
                    cv2.imshow("video",img0)
                    key=cv2.waitKey(1)
               
                t3=0 
                print(flag_msk)
                
                if flag_msk==1:
                    GPIO.setup(servo_SIG, GPIO.OUT)
                    servo = GPIO.PWM(servo_SIG, servo_freq)
                    servo.start(7.5)
                    time.sleep(2)
                    
                    for dc in range(90, -1, -1):
                        dc_trans = mlx.servo_map(dc, 0, 180, servo_width_min, servo_width_max)
                        servo.ChangeDutyCycle(dc_trans)
                        time.sleep(servo_time)
                        
                    while t3<7:
                        t1=time.time()
                        img0 = mlx.cv2AddChineseText(img1,"体温:%.1f\n请尽快通行"%(temp), (0, 160), (0, 255, 0), 60)
                        cv2.imshow("video",img0)
                        key=cv2.waitKey(1)
                        t2=time.time()
                        t3=t3+(t2-t1)

                    for dc in range(1, 91, 1):
                        dc_trans = mlx.servo_map(dc, 0, 180, servo_width_min, servo_width_max)
                        servo.ChangeDutyCycle(dc_trans)
                        time.sleep(servo_time)
                        
                    t3 = 0
                    flag_msk = 0
                            
                elif flag_msk==2:               
                    while True:
                        GPIO.output (pin_sig,GPIO.HIGH)
                        t1=time.time()
                        img0 = mlx.cv2AddChineseText(img1,"体温:%.1f\n请规范佩戴口罩"%(temp), (0, 160), (255, 255, 0), 60)
                        cv2.imshow("video",img0)
                        key=cv2.waitKey(1)
                        t2=time.time()
                        t3=t3+(t2-t1)
                        if t3>2:
                            t3=0
                            flag_msk=0
                            break
                 
                elif flag_msk==3:
                    while True:
                        GPIO.output (pin_sig,GPIO.HIGH)
                        t1=time.time()
                        img0 = mlx.cv2AddChineseText(img1,"体温:%.1f\n请佩戴口罩"%(temp), (0, 160), (255, 0, 0), 60)
                        cv2.imshow("video",img0)
                        key=cv2.waitKey(1)
                        t2=time.time()
                        t3=t3+(t2-t1)
                        if t3>2:
                            t3=0
                            flag_msk=0
                            break
                            
            elif temp>37.5 and temp<43.5:
                GPIO.output (pin_sig,GPIO.HIGH)
                t1=time.time()
                img0 = mlx.cv2AddChineseText(img1,"体温:%.1f\n疑似发烧,请走专用通道"%(temp), (0, 160), (255, 0, 0), 60)
                cv2.imshow("video",img0)
                key=cv2.waitKey(1)
                t2=time.time()
                t3=t3+(t2-t1)
                if t3>3:
                    t3=0
                
            
    except KeyboardInterrupt:
        pass
    cap.release()
    servo.stop()  # 停止pwm
    GPIO.cleanup()  # 清理GPIO引脚





