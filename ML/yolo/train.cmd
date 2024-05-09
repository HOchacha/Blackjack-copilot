set data="C:\Blackjack-copilot\ML\yolo\datasets\Playing Cards.v4-fastmodel-resized640-aug3x.yolov8\data.yaml"
yolo detect train data=%data% model=yolov8n.pt device=0
if %ERRORLEVEL%==0 echo done
pause
