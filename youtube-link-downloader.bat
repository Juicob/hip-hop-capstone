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
cd D:\Python_Projects\flatiron\class-materials\capstone-audio\jayz\reasonable-doubt

youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=8P12bMQMXLA&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=1
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=pUGLfN4oSCg&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=2
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=gwbUtfEJ8lE&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=3
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=p3grvCbdONk&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=4
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=2MggS4f-Puc&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=5
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=xjEoo14xHRs&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=6
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=u4FKKegTGFU&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=7
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=auWvT_rwRPU&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=8
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=YZ_LRnj-2fI&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=9
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=JEYcr5iMjm0&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=10
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=3HjubPJ-Xjk&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=11
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=Vf2LNPFLPAE&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=12
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=AGIsPZFJ4PU&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=13
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=uABZCd1O08c&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=14
timeout /t 5
youtube-dl -x --audio-format "m4a" https://www.youtube.com/watch?v=yBQ210avlx4&list=PLJ808X3foyC7q4qwNDCNuuQL7m-4natMj&index=15


pause
exit