@echo off
title Mouse Forwarder
cd /d "%~dp0pc_app\electron"
npx electron .
exit