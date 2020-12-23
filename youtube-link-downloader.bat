@ECHO ON
TITLE Youtube Downloader


SET mm=-%date:~4,2%-
SET dd=%date:~7,2%-
SET yy=%date:~12,2%

SET hh=%time:~0,2%`
SET min=%time:~3,2%`
SET ss=%time:~6,2%

call C:\ProgramData\Anaconda3\Scripts\activate.bat
call conda activate tensorflow
cd d:
cd D:\Python_Projects\flatiron\class-materials\capstone-audio\jayz\the-blueprint

youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=4PE2mLFp1uU
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=h2fNacQU1jY
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=NS9WS0v8Qpk
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=rwO0aui0sdM
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=0yQlwwt_CSs
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=E0BuFm3KaqU
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=w5srnNrICJo





exit