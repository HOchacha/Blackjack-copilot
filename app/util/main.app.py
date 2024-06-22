import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

fabspath = os.path.abspath(__file__)
yoluster_dir = os.path.join(fabspath, "..", "..", "..", "ML")
sys.path.append(yoluster_dir)

from recommend import *
from yoluster import YOLOCluster  # YOLOCluster 클래스가 포함된 모듈

class BlackjackUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Blackjack Copilot")
        self.setGeometry(100, 100, 1280, 720)

        self.model = YOLOCluster()

        self.initUI()

        self.source = 0
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def initUI(self):
        self.dealer_label = QLabel("Dealer Cards", self)
        self.dealer_label.setGeometry(50, 20, 200, 30)

        self.dealer_card1 = QLabel(self)
        self.dealer_card1.setGeometry(50, 50, 100, 150)
        self.dealer_card2 = QLabel(self)
        self.dealer_card2.setGeometry(150, 50, 100, 150)

        self.your_label = QLabel("Your Cards", self)
        self.your_label.setGeometry(50, 220, 200, 30)

        self.your_card1 = QLabel(self)
        self.your_card1.setGeometry(50, 250, 100, 150)
        self.your_card2 = QLabel(self)
        self.your_card2.setGeometry(150, 250, 100, 150)

        self.recommendation_label = QLabel("Recommended Action", self)
        self.recommendation_label.setGeometry(400, 20, 300, 30)

        self.action_label = QLabel("", self)
        self.action_label.setGeometry(400, 50, 300, 150)
        self.action_label.setStyleSheet("font-size: 24px; color: black")

        self.video_label = QLabel(self)
        self.video_label.setGeometry(400, 250,800, 450)

    def update_frame(self):
        success, frame = self.cap.read()

        if success:
            results = self.model(frame)
            result = results[0]
            print("read")
            mparr = result.mparr
            if len(mparr) > 1:
                dealer_upcard = mparr[0][0]
                print(dealer_upcard)
                if len(mparr[1]) > 1:
                    print("read")

                    recommended_action = get_recommended_action(mparr[1], dealer_upcard)
                    action_text = self.get_action_text(recommended_action)
                    self.action_label.setText(action_text)

                    self.update_cards(mparr)

            annotated_frame = self.model.plotc(result)
            self.display_video(annotated_frame)
        else:
            self.cap.release()

    def get_action_text(self, action_code):
       
        if action_code == -1:
            return self.action_label.text()
        actions = {
            -1: "No Action",
            0: "Stand",
            1: "Hit",
            2: "Double",
            4: "Split",
            8: "Surrender"
        }
        return actions.get(action_code, "Unknown")

    def update_cards(self, mparr):
        dealer_cards = mparr[0]
        your_cards = mparr[1]

        self.dealer_card1.setText(dealer_cards[0] if len(dealer_cards) > 0 else "")
        self.dealer_card2.setText(dealer_cards[1] if len(dealer_cards) > 1 else "")

        self.your_card1.setText(your_cards[0] if len(your_cards) > 0 else "")
        self.your_card2.setText(your_cards[1] if len(your_cards) > 1 else "")

    def display_video(self, frame):
        qformat = QImage.Format_RGB888
        img = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        img = img.rgbSwapped()

        self.video_label.setPixmap(QPixmap.fromImage(img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlackjackUI()
    window.show()
    sys.exit(app.exec_())
