from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import time
from datetime import date
import sys
import pandas as pd
import json


options = webdriver.ChromeOptions()
options.add_argument ("lang = en_us")
options.page_load_strategy = 'eager'
# options.add_argument("--headless")
driver = webdriver.Chrome(executable_path='chromedriver.exe',options = options)
driver.set_window_position(-1500, 100)
driver.set_window_size(1000, 1000)

start_time = time.time()

def csss(parent, sel):
    return parent.find_elements_by_css_selector(sel)
def css(parent, sel):
    return parent.find_element_by_css_selector(sel)
def xpaths(parent, sel):
    return parent.find_elements_by_xpath(sel)
def xpath(parent, sel):
    return parent.find_element_by_xpath(sel)



def make_album_row():
    album_row = {}
    album_row['album_title'] = None
    album_row['album_url'] = None
    album_row['album_cover'] = None
    album_row['artist'] = None
    album_row['alternate_names'] = None
    return album_row   
    
albums = []
def scrape_albums():
    # print(len(csss(driver, 'div.profile_list_item mini-album-card')))
    show_albums_selector = "//div[contains(@class, 'full_width_button')][contains(text(),' albums ')]"
    driver.find_element_by_xpath(show_albums_selector).click()
    sleep(1)
    for album in csss(driver, 'div.profile_list_item mini-album-card'):
        album_row = make_album_row()
        album_row['album_url'] = css(album, 'a').get_attribute('href')
        album_row['album_title'] = css(album, 'a').get_attribute('title')
        album_row['album_cover'] = css(album, 'a div.mini_card-thumbnail').get_attribute('style').split('"')[1]
        album_row['artist'] = css(driver, 'h1').text.strip()
        try:
            album_row['alternate_names'] = css(driver, 'div.profile_identity-alternate_names').text.strip()
        except:
            print('No alternate artist name')
        print(album_row)
        albums.append(album_row)
    return albums
                        # ? I'm thinkin I'll have it first run through to gather albums then songs titles, then lyrics for each - 11/6/2020
                        
# def try_hard(key, value):
#     ''' Wrap the world in try/excepts
#         I'm so tired of running into random ass elements that don't have things that it should
#         so it's defaulting to none and if it's there it'll be filled - this could be doc'd better
#         but this is the mood atm - but wooooww idek how to write this write now...'''
        
#     tryddd:
#         key = value
#         return ....
#     except:
#         print(f'tried to find {key} with {value} but nah')


def make_track_row():
    track_row = {}
    track_row['album'] = None
    track_row['track_title'] = None
    track_row['track_url'] = None
    track_row['track_views'] = None
    return track_row

tracks = []
def scrape_tracks():
    for track in csss(driver, 'div.chart_row'):
        track_row = make_track_row()
        track_row['album'] =  css(driver, 'h1.header_with_cover_art-primary_info-title').text
        track_row['track_title'] = css(track, 'h3').text.strip().replace(' Lyrics','')
        try:
            track_row['track_url'] = css(track, 'a').get_attribute('href')
        except:
            print(f'tried to find track_url')
        try:
            track_row['track_views'] = css(track, 'div.chart_row-metadata_element').text.strip()
        except:
            print(f'tried to find track_views')
        # todo consider standardizing view values (thousands vs millions) or keep numbers only - 11/6/2020
        # todo and filter by < small number to denote views in millions as hits - 11/6/2020
        print(track_row)
        tracks.append(track_row)
    return tracks

def make_lyrics_row():
    lyrics_row = {}
    lyrics_row['album'] = None
    lyrics_row['lyrics_title'] = None
    lyrics_row['lyrics_url'] = None
    lyrics_row['lyrics'] = None
    lyrics_row['track_views'] = None
    lyrics_row['feature'] = None
    lyrics_row['producer'] = None
    return lyrics_row

lyrics = []
def scrape_lyrics():
    for track in tracks:
        lyrics_row = make_lyrics_row()
        try:
            driver.get(track['track_url'])
        except:
            continue
        # adding pauses to not be a dick
        sleep(1.5)
        # Not sure if this is really it or not but pages sometimes seem to get stuck until you scroll down a bit so adding a few page downs seems to move things along
        ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()
        ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()
        ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()
        lyrics_row['album'] = track['album']
        lyrics_row['lyrics_url'] = track['track_url']
        lyrics_row['lyrics_title'] = track['track_title']
        lyrics_row['track_views'] = track['track_views']
        # there are one or two tracks that don't actually have lyrics 
        try:
            lyrics_row['lyrics'] = css(driver, 'section p').text
            print('First few words of lyrics page are: ', lyrics_row['lyrics'].split(' ')[:5])
        except:
            print('-- No lyrics found --')
        try:
            lyrics_row['feature'] = css(driver, 'h3 expandable-list[collection="song.featured_artists"] > div"] > div').text.strip()
            print(lyrics_row['feature'])
        except:
            print('-- No featured artists found --')
        try:
            lyrics_row['producer'] = css(driver, 'h3 expandable-list[collection="song.producer_artists"] > div').text.strip()
            print(lyrics_row['producer'])
        except:
            print('-- No producer found --')
        
        # added to view progress while the scraper is running
        print(lyrics_row['lyrics_title'])
        print(f'{len(lyrics) + 1} / {len(tracks)}')
        print()
        lyrics.append(lyrics_row)
        # could probably save this for the end so it's not writing every iteration but I had some issues with errors at first so I put it here and think I'll just leave it for now

    return lyrics

# set source urls
artist_urls = [
    'https://genius.com/artists/Jay-z',
    # 'https://genius.com/artists/Outkast',
    # 'https://genius.com/artists/Ugk',
    # 'https://genius.com/artists/The-notorious-big',
    # 'https://genius.com/artists/Nas',
    # 'https://genius.com/artists/Yasiin-bey',
    # 'https://genius.com/artists/Kanye-west',
    # 'https://genius.com/artists/common',
    # 'https://genius.com/artists/eminem',
    # 'https://genius.com/artists/j-dilla',
    # 'https://genius.com/artists/lil-wayne',
    # 'https://genius.com/artists/ti',
    # 'https://genius.com/artists/2pac',
    # 'https://genius.com/artists/snoop-dogg',
    # 'https://genius.com/artists/too-short',
    # 'https://genius.com/artists/dr-dre',
    # 'https://genius.com/artists/ice-cube',
    # 'https://genius.com/artists/chance-the-rapper',
    # 'https://genius.com/artists/lupe-fiasco',
    # 'https://genius.com/artists/mac-miller',
    # 'https://genius.com/artists/royce-da-59',
    # 'https://genius.com/artists/nelly',
    # 'https://genius.com/artists/childish-gambino',
    # 'https://genius.com/artists/future',
    # 'https://genius.com/artists/travis-scott',
    # 'https://genius.com/artists/j-cole',
    # 'https://genius.com/artists/Rick-Ross',
    # 'https://genius.com/artists/2-Chainz',
    # 'https://genius.com/artists/Ludacris',
    # 'https://genius.com/artists/Jeezy',
    # 'https://genius.com/artists/Tyler-the-Creator',
    # 'https://genius.com/artists/E-40',
    # 'https://genius.com/artists/The-Game',
    # 'https://genius.com/artists/Nipsey-Hussle',
    # 'https://genius.com/artists/Warren-G',


    ]

# for album in album_urls:
#     driver.get(album)
#     scrape_tracks()
# # using the scraped links from the albums, go to each track and grab lyrics
# scrape_lyrics()

for artist in artist_urls:
    driver.get(artist)
    # go to all album links and pull information, mainly album urls for track links
    scrape_albums()
# saving album data to csv in case i need to use the data separately
albums_df = pd.DataFrame(albums)
print(albums_df.head())
main_path = r'D:\Python_Projects\flatiron\class-materials\hip-hop-capstone\lyrics\\'
date_stamp_and_file_ext = str(time.strftime("%Y%m%d-%H%M%S")) + '.csv'

albums_df.to_csv(f'{main_path}albums{date_stamp_and_file_ext}', index = False, encoding='utf-8-sig')

# go to each album gathered from each artist and pull information, mainly track links for lyrics
for album in albums[:2]:
    driver.get(album['album_url'])
    scrape_tracks()

# saving tracks data to csv in case i need to use the data separately
tracks_df = pd.DataFrame(tracks)
tracks_df.to_csv(f'{main_path}tracks{date_stamp_and_file_ext}', index = False, encoding='utf-8-sig')
print(tracks_df.head())

# go to each track link and pull information, mainly lyrics
scrape_lyrics()

# saving lyrics data
lyrics_df = pd.DataFrame(lyrics)
lyrics_df.to_csv(f'{main_path}lyrics{date_stamp_and_file_ext}', index = False, encoding='utf-8-sig')

# just a few quality of life things for myself for when the program finishes
print(lyrics_df.head())
try:
    print(f'{lyrics_df["lyrics"].count()} items scraped')
except:
    pass
print('Exporting df to csv')
print(f'Saved to - {main_path}')
print("\nDope Genius Scraper took", (time.time() - start_time)/60, "minutes to run\n")
print('Closing in 3 seconds')
    
sleep(3)
driver.quit()
sys.exit()