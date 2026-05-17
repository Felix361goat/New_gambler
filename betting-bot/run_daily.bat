@echo off
cd /d "C:\Users\felix\Desktop\new_gambler\betting-bot"

echo [%date% %time%] Starting betting bot... >> logs\daily.log

python main.py --collect >> logs\daily.log 2>&1
python main.py --predict >> logs\daily.log 2>&1
python main.py --brief >> logs\daily.log 2>&1

echo [%date% %time%] Done. >> logs\daily.log
