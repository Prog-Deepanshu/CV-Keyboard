import cv2
import numpy as np
from time import sleep
from pynput.keyboard import Controller
from HandTrackingModule import handDetector
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = handDetector(detectionCon=0.8)
keyboard = Controller()

keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["<", " "]
]

finalText = ""

class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = [Button([100 * j + 50, 100 * i + 50], key) for i, row in enumerate(keys) for j, key in enumerate(row)]

def draw_all(img, button_list):
    img_new = np.zeros_like(img, np.uint8)
    for button in button_list:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img_new, (x, y, w, h), 20, rt=0)
        cv2.rectangle(img_new, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img_new, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    alpha = 0.5
    mask = img_new.astype(bool)
    out = cv2.addWeighted(img, alpha, img_new, 1 - alpha, 0)
    out[mask] = img_new[mask]
    return out

def is_finger_in_button(lm_list, button):
    x, y = button.pos
    w, h = button.size
    return x < lm_list[8][1] < x + w and y < lm_list[8][2] < y + h

def is_pinch_gesture(lm_list):
    l, _, _ = detector.findDistance(8, 12, img, draw=False)
    return l < 40

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lm_list, _ = detector.findPosition(img)

    img = draw_all(img, buttonList)

    if lm_list:
        for button in buttonList:
            if is_finger_in_button(lm_list, button):
                cv2.rectangle(img, button.pos, (button.pos[0] + button.size[0], button.pos[1] + button.size[1]), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (button.pos[0] + 20, button.pos[1] + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                if is_pinch_gesture(lm_list):
                    cv2.rectangle(img, button.pos, (button.pos[0] + button.size[0], button.pos[1] + button.size[1]), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (button.pos[0] + 20, button.pos[1] + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    if button.text == "<":
                        finalText = finalText[:-1]
                        keyboard.press('\010')
                    else:
                        finalText += button.text
                        keyboard.press(button.text)
                    sleep(0.1)

    cv2.rectangle(img, (50, 710), (700, 610), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 690), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
