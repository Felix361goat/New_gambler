@echo off
echo Richte Windows Task-Scheduler ein...

:: Logs-Ordner erstellen
mkdir "C:\Users\felix\Desktop\new_gambler\betting-bot\logs" 2>nul

:: Morgens um 06:00 - Daten sammeln + Vorhersage + Briefing
schtasks /create /tn "BettingBot-Morning" /tr "\"C:\Users\felix\Desktop\new_gambler\betting-bot\run_daily.bat\"" /sc daily /st 06:00 /f
echo ✅ Morgen-Task erstellt (06:00 Uhr)

:: Abends um 23:00 - Zusammenfassung
schtasks /create /tn "BettingBot-Evening" /tr "\"C:\Users\felix\Desktop\new_gambler\betting-bot\run_evening.bat\"" /sc daily /st 23:00 /f
echo ✅ Abend-Task erstellt (23:00 Uhr)

echo.
echo Fertig! Der Bot startet ab morgen automatisch.
echo Morgens um 06:00 bekommst du die Wetten auf Telegram.
pause
