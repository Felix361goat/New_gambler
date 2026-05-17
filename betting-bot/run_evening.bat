@echo off
cd /d "C:\Users\felix\Desktop\new_gambler\betting-bot"

echo [%date% %time%] Evening summary... >> logs\daily.log

python main.py --odds_snapshot >> logs\daily.log 2>&1
python main.py --results >> logs\daily.log 2>&1
python main.py --summarize >> logs\daily.log 2>&1

echo [%date% %time%] Evening done. >> logs\daily.log
