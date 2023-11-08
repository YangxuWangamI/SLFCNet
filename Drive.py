from ultralytics import YOLO

def Task():
    # ------------------------------------------------------------------------------------------------------------------
    task = 'train'
    run_model = r'runs/detect/train/weights/last.pt'
    # ------------------------------------------------------------------------------------------------------------------
    if task == 'train':
        model = YOLO('ultralytics/cfg/models/SLFCNet.yaml').load('yolov8n.pt')
        model.train(
            device='0',
            imgsz=640,
            data='ultralytics/cfg/datasets/train.yaml',
            epochs=450,
            batch=16,
            cache=True,
            workers=2,
            resume=False,
        )
    elif task == 'val':
        model = YOLO(run_model)
        model.val(
            imgsz=640,
            data='ultralytics/cfg/datasets/val.yaml',
            conf=0.001,
            iou=0.5,
            batch=1,
            task='test',
        )
    elif task == 'predict':
        model = YOLO(run_model)
        model.predict(
            source=r'img',
            imgsz=640,
            save=True,  # save or not save
            conf=0.5,
            iou=0.5,
            line_width=6,
            show=False,  # show images
            max_det=800,
            save_txt=False,
            show_conf=True,  # show conf
            show_labels=True,  # show label
        )
if __name__ == '__main__':
    Task()