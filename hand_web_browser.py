import cv2
import numpy as np
import math
import webbrowser as wb
import os

print("Enter full website for")
print("\n2 fingers")
fingers2 = input()
print("\n3 fingers")
fingers3 = input()
print("\n4 fingers")
fingers4 = input()

tabs = 0
count = 0
cap = cv2.VideoCapture(0)
website_opened = False

while cap.isOpened():
    # read image
    ret, img = cap.read()

    # Adjusted rectangle coordinates
    top_left = (200, 200)  # Adjust these values to fit your hand
    bottom_right = (35, 35)  # Adjust these values to fit your hand
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 0)
    crop_img = img[bottom_right[1]:top_left[1], bottom_right[0]:top_left[0]]

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    blurred = cv2.GaussianBlur(grey, (15, 15), 0)  # Adjust the kernel size

    # thresholding: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # show thresholded image, not necessary and can be skipped
    cv2.imshow('Thresholded', thresh1)

    # check OpenCV version to avoid unpacking error
    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find contour with max area
    cnt = max(contours, key=lambda x: cv2.contourArea(x), default=None)

    if cnt is not None:
        # create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defectsface
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # vectors representing fingers
            vec1 = np.array(start) - np.array(far)
            vec2 = np.array(end) - np.array(far)

            # calculate the angle using the dot and cross products
            angle_cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.degrees(np.arccos(angle_cosine))

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)

        if count == 0:
            cv2.putText(img, "Wait for it :p", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)

        # define actions required
        if count_defects == 1 and count != 2 and tabs <= 8 and not website_opened:
            wb.open_new_tab('http://www.' + fingers2 + '.com')
            tabs = tabs + 1
            cv2.putText(img, "2." + fingers2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
            count = 2
            website_opened = True
        elif count_defects == 2 and count != 3 and tabs <= 8 and not website_opened:
            wb.open_new_tab('http://www.' + fingers3 + '.com')
            tabs = tabs + 1
            cv2.putText(img, "3." + fingers3, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            count = 3
            website_opened = True
        elif count_defects == 3 and count != 4 and tabs <= 8 and not website_opened:
            wb.open_new_tab('http://www.' + fingers4 + '.com')
            cv2.putText(img, "4." + fingers4, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 165, 0), 3)
            tabs = tabs + 1
            count = 4
            website_opened = True
        elif count_defects == 4 and count != 5:
            cv2.putText(img, "5.Close Web browser", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
            os.system("taskkill /im chrome.exe /f")
            tabs = 0
            count = 5
            website_opened = False
        else:
            cv2.putText(img, "", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)

        if count == 2:
            cv2.putText(img, "2." + fingers2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
        elif count == 3:
            cv2.putText(img, "3." + fingers3, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        elif count == 4:
            cv2.putText(img, "4." + fingers4, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 165, 0), 3)
        elif count == 5:
            cv2.putText(img, "5.WebBrowser close", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)

        # show appropriate images in windows
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        # not necessary to show contours and can be skipped
        cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
