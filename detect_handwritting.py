# file: detect_text_regions.py

import cv2
from mmocr.utils.ocr import MMOCR

def detect_text_regions(image_path: str, output_path: str = None):
    # Initialize MMOCR detector (choose model: e.g. DBNet, TextSnake, etc.)
    ocr = MMOCR(det='DBNet', recog=None)  # only detection

    # Run detection
    results = ocr.readtext(image_path)

    # results: list of dicts, each with 'bbox' (polygon or rotated box), etc.
    img = cv2.imread(image_path)

    for item in results:
        bbox = item['bbox']  # usually list of 4 or more (x, y) points
        # convert to int
        pts = [(int(x), int(y)) for x, y in bbox]

        # draw polygon
        pts_arr = cv2.array(pts).reshape((-1,1,2))
        cv2.polylines(img, [pts_arr], isClosed=True, color=(0,255,0), thickness=2)

    # Save or show
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow('detected', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    INPUT = 'test3.jpeg'   # <-- your BOL image path
    OUTPUT = 'bol_doc_detected.jpg'
    detect_text_regions(INPUT, OUTPUT)
    print('Done — saved to', OUTPUT)
