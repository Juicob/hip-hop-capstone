from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
from datetime import date
import sys
import pandas as pd
import json


options = webdriver.ChromeOptions()
options.add_argument ("lang = en_us")
options.page_load_strategy = 'eager'
# options.add_argument("--headless")
driver = webdriver.Chrome(executable_path='chromedriver.exe',options = options)
driver.set_window_size(800, 600)
driver.set_window_position(-1500, 100)

def csss(parent, sel):
    return parent.find_elements_by_css_selector(sel)
def css(parent, sel):
    return parent.find_element_by_css_selector(sel)
def xpaths(parent, sel):
    return parent.find_elements_by_xpath(sel)
def xpath(parent, sel):
    return parent.find_element_by_xpath(sel)


def make_track_row():
    track_row = {}
    track_row['album'] = None
    track_row['track_title'] = None
    track_row['track_url'] = None
    return track_row

tracks = []
def scrape_tracks():
    for track in csss(driver, 'div#contents ytd-playlist-video-renderer'):
        track_row = make_track_row()
        track_row['album'] =  css(driver, 'h1#title a').text
        track_row['track_title'] = css(track, 'h3 span#video-title').text.strip()
        try:
            track_row['track_url'] = css(track, 'div#content > a').get_attribute('href')
        except:
            print(f'tried to find track_url')
        print(track_row)
        tracks.append(track_row)
    return tracks

playlist_urls = [
    'https://www.youtube.com/playlist?list=PL61YjVa1b9jkCMdfftZt9GbnGVwfyXuTP',
    'https://www.youtube.com/playlist?list=PLr1fYOvQFfPfHp8f_RhLjTBFqaU0lRSF_&pbjreload=102',
    'https://www.youtube.com/playlist?list=PLSwyLbwxPYdBGRoAhaIxs8b_OAuF5bGPB',

    ]

# go to all album links and scrape information
for album in playlist_urls:
    driver.get(album)
    scrape_tracks()


tracks_df = pd.DataFrame(tracks)

# just a few quality of life things for myself for when the program finishes
print(tracks)
print(tracks_df.head())
print(f'{tracks_df["track_title"].count()} items scraped')
print('Exporting df to csv')

save_path = r'D:\Python_Projects\flatiron\class-materials\hip-hop-capstone\youtube-link-scraper\jayz-yt-links' + str(date.today()).replace('-','') + '.csv'
print(f'Saved to - {save_path}')
tracks_df.to_csv(f'{save_path}', index = False, encoding='utf-8-sig')
    
sleep(3)
print('Closing')
driver.quit()
sys.exit()